# Standard library
import streamlit.components.v1 as components

import copy
import csv
import itertools
import math
import os
import subprocess
import sys
import time
import importlib
from typing import List

# Third-party libraries
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

# Local modules
sys.path.append('..')
sys.path.append('main')

from envs.arterial_multi_lanes import ArterialMultiLanes
from lib.car_dimensions import CarDimensions, BicycleModelDimensions, BicycleRealDimensions
from lib.collision_avoidance import get_cutoff_curve_by_position_idx, check_collision_moving_bicycle
from lib.motion_primitive import load_motion_primitives
from lib.mp_search_reasoning import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleArterial
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car, draw_bicycle, draw_astar_search_points
from lib.reasons_evaluation import evaluate_distance_to_centerline, evaluate_distance_to_obstacle, \
    evaluate_time_following
from lib.simulation import History, HistorySimulation, Simulation, State
from lib.trajectories import calc_nearest_index_in_direction, resample_curve
# In overtaking_cyclist_bidirectional_road.py
from lib.parameters import CyclistParameters, DriverParameters, ScenarioParameters, MPCParameters, ReasonParameters

# Initialize logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist, ideal_weight_driver,
         ideal_weight_policymaker, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2, yyy3, zzz0, zzz1, zzz2, zzz3,
         replanner: bool = False, vis_frame: bool = False, save_weight_table: bool = False,
         historical_plot: bool = False, save_path=None) -> None:
    """
    Main function to simulate the scenario of an AV overtaking a cyclist in a bidirectional road.

    Args:
        replanner (bool): If True, enables replanner mode for the simulation.
        save_path (Path): The Path object for the directory to save frames and results.
    """

    # Initialization
    TIME_ELAPSED_DRIVER = 0  # time elapsed since the driver followed the bicycle at a distance less than DriverParameters.DISTANCE_REF
    TIME_PASSED_CYCLIST = 0  # time elapsed since the driver followed the bicycle at a distance less than CyclistParameters.DISTANCE_REF
    IS_FOLLOWING = True  # Flag to determine if the driver will be following the vehicle in front

    # Lists to store simulation data
    reasons_policymaker_values = []
    reasons_driver_values = []
    reasons_cyclist_values = []
    distance_values = []
    speed_values = []
    xref_deviation_values = []
    time_values = []

    # A variable that saves information whether a reason has already triggered a replan
    replan_tracker = False

    # The save folder is now passed as an argument.
    # We no longer need to define it here using relative paths.

    # Check if a save path was provided
    if save_path is None:
        raise ValueError("A save path must be provided to the main function.")

    # Ensure the folder exists (create it if it doesn't)
    # This is handled in the run_simulation function, but it's good practice to have it here too
    save_path.mkdir(parents=True, exist_ok=True)

    # Delete all existing .jpg files in the folder using the passed-in Path object
    for file in save_path.glob("*.jpg"):
        os.remove(file)

    # Initialize simulation
    mps, car_dimensions, bicycle_dimensions, arterial, scenario_no_obstacles, scenario_visualization, moving_obstacles = initialize_simulation()

    # Run motion primitive search
    cost, path, trajectory_full, search_runtime = run_motion_primitive_search(scenario_no_obstacles, car_dimensions,
                                                                              mps, xxx0=xxx0, xxx1=xxx1, xxx2=xxx2,
                                                                              xxx3=xxx3, yyy0=yyy0, yyy1=yyy1,
                                                                              yyy2=yyy2, yyy3=yyy3, zzz0=zzz0,
                                                                              zzz1=zzz1, zzz2=zzz2, zzz3=zzz3)

    # Initialize MPC, set max speed to cyclist speed if the AV is following the cyclist
    if IS_FOLLOWING == True:
        mpc, state, dl = initialize_mpc(trajectory_full, car_dimensions, max_speed=CyclistParameters.SPEED)
    else:
        mpc, state, dl = initialize_mpc(trajectory_full, car_dimensions, max_speed=MPCParameters.MAX_SPEED_FREEWAY)

    simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=ScenarioParameters.DT,
                                   initial_state=state)
    history = simulation.history  # gets updated automatically as simulation runs

    # Number of index to cut off the trajectory before a collision occurs, change 2 for the larger index cut off
    EXTRA_CUTOFF_MARGIN = 2 * int(
        math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length

    traj_agent_idx = 0
    tmp_trajectory = None

    loop_runtimes = []

    # Create a list to store the location of obstacles through time for visualization
    obstacles_positions = [[] for _ in moving_obstacles]

    start_time = time.time()
    for i in itertools.count():
        loop_start_time = time.time()
        if mpc.is_goal(state):
            break

        # Update trajectory index to the nearest future index in the trajectory based on the current state
        traj_agent_idx = update_trajectory_index(state, tmp_trajectory, traj_agent_idx, trajectory_full)

        # Get the trajectory from the current index to the end, removing the points that have already passed
        trajectory_res = trajectory = trajectory_full[traj_agent_idx:]

        # Compute the predicted trajectory for the car
        trajectory_res = compute_predicted_trajectory(state, trajectory_res)

        # Predict the movement of each moving obstacle, and retrieve the predicted trajectories
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=ScenarioParameters.DT,
                                                car_dimensions=bicycle_dimensions)
                      .state_prediction(MPCParameters.TIME_HORIZON)).T
            for o in moving_obstacles]

        # Evaluate reasons
        reasons_policymaker_reg_compliance, reasons_driver_time_eff, reasons_cyclist_comfort, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST = evaluate_reasons(
            state, moving_obstacles, car_dimensions, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST)

        # Find the collision location
        collision_xy = check_collision_moving_bicycle(car_dimensions, bicycle_dimensions, trajectory_res, trajectory,
                                                      trajs_moving_obstacles,
                                                      frame_window=MPCParameters.FRAME_WINDOW)

        # If replanner is enabled, based on reasons evaluation, determine if a replan is needed
        if replanner == True:
            # Flag to determine if a replan should happen
            replan_needed = False

            # Evaluate reasons and determine if a replan is needed, replan_tracker is used to prevent multiple replans
            replan_needed, replan_tracker = reasons_evaluation(reasons_cyclist_comfort, reasons_driver_time_eff,
                                                               reasons_policymaker_reg_compliance, replan_needed,
                                                               replan_tracker)

            # Execute replan only if reasons value drop below ScenarioParameters.REASONS_THRESHOLD was detected
            if replan_needed:
                # Perform replan, reset the trajectory, MPC, and collision status
                IS_FOLLOWING = False
                # change max_speed of the MPC to 30/3.6
                collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles = perform_replan(
                    weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist, ideal_weight_driver,
                    ideal_weight_policymaker, arterial, car_dimensions, bicycle_dimensions, dl, moving_obstacles, mps,
                    state,
                    trajs_moving_obstacles, scenario_visualization,
                    reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                    reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values,
                    time_values, xxx0=xxx0, xxx1=xxx1, xxx2=xxx2, xxx3=xxx3, yyy0=yyy0, yyy1=yyy1, yyy2=yyy2, yyy3=yyy3,
                    zzz0=zzz0, zzz1=zzz1, zzz2=zzz2, zzz3=zzz3,
                    max_speed=MPCParameters.MAX_SPEED_FREEWAY,
                    is_following=IS_FOLLOWING,
                    vis_frame=vis_frame,
                    save_weight_table=save_weight_table,
                    time_elapsed_driver=TIME_ELAPSED_DRIVER,
                    time_passed_cyclist=TIME_PASSED_CYCLIST
                )
                scenario = scenario_obstacles

        # Cut off the trajectory before a collision occurs, with an additional margin
        if collision_xy is not None:
            tmp_trajectory = cutoff_trajectory_before_collision(EXTRA_CUTOFF_MARGIN, collision_xy, traj_agent_idx,
                                                                trajectory_full)
        else:
            tmp_trajectory = trajectory_full

        # Pass the cut trajectory to the MPC
        mpc.set_trajectory_fromarray(tmp_trajectory)

        # Compute the MPC
        delta, acceleration = mpc.step(state)

        # Runtime calculation
        loop_end_time = time.time()
        loop_runtime = loop_end_time - loop_start_time
        loop_runtimes.append(loop_runtime)
        xref_deviation_value = mpc.get_current_xref_deviation()
        # show the computation results
        if vis_frame == True:
            # CRITICAL: Pass the save_path variable to visualize_frame()
            visualize_frame(ScenarioParameters.DT, car_dimensions, bicycle_dimensions, collision_xy, i,
                            moving_obstacles, mpc,
                            scenario_visualization, simulation,
                            state, tmp_trajectory, trajectory_res,
                            reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                            reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                            speed_values, time_values, xref_deviation_values, xref_deviation_value,
                            static_x_axis=True, max_time=15, historical_plot=False, save_path=save_path)

        # Move all obstacles one step ahead
        for i_obs, o in enumerate(moving_obstacles):
            obstacles_positions[i_obs].append((i, o.get()))  # i is time here
            o.step()

        # Step the simulation (i.e. move our agent forward)
        state = simulation.step(a=acceleration, delta=delta, xref_deviation=mpc.get_current_xref_deviation())

    # Print runtimes
    end_time = time.time()
    loops_total_runtime = sum(loop_runtimes)
    total_runtime = end_time - start_time
    # logger.info('total loops run time is: {}'.format(loops_total_runtime))
    # logger.info('total run time is: {}'.format(total_runtime))
    # logger.info('each mpc runtime is: {}'.format(loops_total_runtime / len(loop_runtimes)))

    # Visualize final
    visualize_final(simulation.history)

    # Save vehicle data
    save_vehicle_data(simulation, time_values, reasons_policymaker_values, reasons_driver_values,
                      reasons_cyclist_values, replanner)

    # Plot Trajectories and Conflicts
    plot_trajectories(obstacles_positions, simulation.history)


def cutoff_trajectory_before_collision(EXTRA_CUTOFF_MARGIN, collision_xy, traj_agent_idx,
                                       trajectory_full) -> np.ndarray:
    """
    Cut off the trajectory before a collision occurs, with an additional margin.

    Args:
        EXTRA_CUTOFF_MARGIN: Additional margin to ensure safety.
        collision_xy: Coordinates of the collision point (x, y), or None if no collision.
        traj_agent_idx: Current index in the trajectory.
        trajectory_full: The full trajectory from the motion primitive search.

    Returns:
        np.ndarray: The truncated trajectory.
    """
    # cutoff the curve such that it ends right before the collision (and some margin)
    cutoff_idx = get_cutoff_curve_by_position_idx(trajectory_full, collision_xy[0],
                                                  collision_xy[1]) - EXTRA_CUTOFF_MARGIN
    cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
    tmp_trajectory = trajectory_full[:cutoff_idx]

    return tmp_trajectory


def compute_predicted_trajectory(state, trajectory_res, last_index=None):
    """
    Compute the predicted trajectory for the car based on its current speed and maximum acceleration.

    Args:
        state: The current state of the car (including velocity).
        trajectory_res: The current trajectory to be resampled.

    Returns:
        np.ndarray: The resampled trajectory based on acceleration.
    """
    if last_index is None:
        if state.v < Simulation.MAX_SPEED:
            resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
            resample_dl = np.cumsum(resample_dl) + state.v
            resample_dl = ScenarioParameters.DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
            trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
        else:
            trajectory_res = resample_curve(trajectory_res, dl=ScenarioParameters.DT * Simulation.MAX_SPEED)
        return trajectory_res
    else:
        trajectory_res = resample_curve(trajectory_res, dl=ScenarioParameters.DT * state.v)
        return trajectory_res


def update_trajectory_index(state, tmp_trajectory, traj_agent_idx, trajectory_full):
    """
    Update the trajectory index based on the current state.

    Args:
        state: The current state of the vehicle.
        tmp_trajectory: The temporary trajectory.
        traj_agent_idx: The current index of the agent in the trajectory.
        trajectory_full: The full trajectory.

    Returns:
        int: The updated index of the agent in the trajectory
    """

    if tmp_trajectory is None or np.any(tmp_trajectory[traj_agent_idx, :] != tmp_trajectory[-1, :]):
        traj_agent_idx = calc_nearest_index_in_direction(state, trajectory_full[:, 0], trajectory_full[:, 1],
                                                         start_index=traj_agent_idx, forward=True)
    return traj_agent_idx


def perform_replan(weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist, ideal_weight_driver,
                   ideal_weight_policymaker, arterial, car_dimensions, bicycle_dimensions, dl, moving_obstacles, mps,
                   state,
                   trajs_moving_obstacles, scenario_visualization,
                   reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                   reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values,
                   time_values, max_speed, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2, yyy3, zzz0, zzz1, zzz2, zzz3,
                   is_following=True, vis_frame=False, save_weight_table=False, time_elapsed_driver=0.0,
                   time_passed_cyclist=0.0):
    """
    Perform a replan based on the current state and moving obstacles.

    Args:
        arterial: The arterial scenario.
        car_dimensions: Dimensions of the car.
        dl: Distance between trajectory points.
        moving_obstacles: List of moving obstacles.
        mps: Motion primitives.
        state: The current state of the vehicle.
        trajs_moving_obstacles: Predicted trajectories of moving obstacles.

    Returns:
        tuple: A tuple containing the updated trajectory, MPC, and collision status.
    """
    # Extract bicycle position and velocity for the motion planner
    bicycle_x = moving_obstacles[0].get()[0]
    bicycle_y = moving_obstacles[0].get()[1]
    bicycle_v = moving_obstacles[0].get()[4]
    bicycle_state = np.array([bicycle_x, bicycle_y, bicycle_v])

    # Create a new scenario with prediction of obstacle positions from current time to the TIME_HORIZON ahead
    scenario_obstacles = arterial.create_scenario(
        moving_obstacles=True,
        moving_obstacles_trajectory=trajs_moving_obstacles,
        spawn_location_x=bicycle_x,
        spawn_location_y=bicycle_y,
        av_location_x=state.x,
        av_location_y=state.y,
        is_following=is_following
    )
    logger.info(f"Initial position: {scenario_obstacles.start}")

    # Perform motion primitive search
    search = MotionPrimitiveSearch(scenario_obstacles, car_dimensions, mps, xxx0=xxx0, xxx1=xxx1, xxx2=xxx2, xxx3=xxx3,
                                   yyy0=yyy0, yyy1=yyy1, yyy2=yyy2, yyy3=yyy3, zzz0=zzz0, zzz1=zzz1, zzz2=zzz2,
                                   zzz3=zzz3, margin=car_dimensions.radius,
                                   moving_obstacles_state=bicycle_state,
                                   driver_elapsed_time=time_elapsed_driver,
                                   cyclist_elapsed_time=time_passed_cyclist
                                   )

    # Calculate all trajectories
    costs, paths, trajectories_full = search.run_all(debug=True)

    # Create a new trajectory for the ego vehicle to follow the cyclist
    follow_trajectory = create_following_trajectory(state, trajectories_full)

    # Add the trajectory to the list of trajectories to the last position
    trajectories_full.append((follow_trajectory, (0.0, 0.0, 0.0, 0.0, 0.0)))
    print("YOYOYOYOYOYO", trajectories_full)
    print('''









    ''')

    if save_weight_table == True:
        # Evaluate trajectories based on human-centered reasons
        weight_table = generate_stakeholder_weight_table(
            trajectories_full=trajectories_full,
            moving_obstacles=moving_obstacles,
            state=state,
            car_dimensions=car_dimensions,
            bicycle_dimensions=bicycle_dimensions,
            reasons_cyclist_comfort=reasons_cyclist_comfort,
            reasons_driver_time_eff=reasons_driver_time_eff,
            reasons_policymaker_reg_compliance=reasons_policymaker_reg_compliance,
            time_elapsed_driver=time_elapsed_driver,
            time_passed_cyclist=time_passed_cyclist,
            weight_step=0.02,  # Adjust for desired granularity
            save_path="stakeholder_weight_analysis.csv"
        )

    agent_weights, eval_results = evaluate_trajectories_for_reasons(
        weightofpolicy,
        weightofdriver,
        weightofcyclist,
        ideal_weight_cyclist,
        ideal_weight_driver,
        ideal_weight_policymaker,
        trajectories_full,
        moving_obstacles,
        state,
        car_dimensions,
        bicycle_dimensions,
        reasons_cyclist_comfort,
        reasons_driver_time_eff,
        reasons_policymaker_reg_compliance,
        time_elapsed_driver=time_elapsed_driver,
        time_passed_cyclist=time_passed_cyclist
    )

    # In perform_replan after evaluation
    if vis_frame == True:
        visualize_trajectory_evaluations(
            eval_results, trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions,
            scenario_visualization, time_values,
            reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, agent_weights,
            save_path=os.path.join("results", "reasons_evaluation", "trajectory_evaluations.png")
        )

    # Use the best trajectory
    best_idx = eval_results['best_idx']
    trajectory_full = trajectories_full[best_idx]
    trajectory_full = trajectory_full[0]

    logger.info(
        f"Selected trajectory {best_idx} with human-centered score: {eval_results['best_evaluation']['total_score']:.3f}")
    logger.info(f"Policymaker: {eval_results['best_evaluation']['avg_scores']['policymaker']:.2f}, "
                f"Driver: {eval_results['best_evaluation']['avg_scores']['driver']:.2f}, "
                f"Cyclist: {eval_results['best_evaluation']['avg_scores']['cyclist']:.2f}")

    # Reset MPC with new trajectory
    traj_agent_idx = 0  # Reset trajectory index
    mpc = MPC(
        cx=trajectory_full[:, 0],
        cy=trajectory_full[:, 1],
        cyaw=trajectory_full[:, 2],
        dl=dl, speed=max_speed, dt=ScenarioParameters.DT, car_dimensions=car_dimensions
    )
    collision_xy = None  # Reset collision tracker

    # Store evaluation data for later analysis if needed
    scenario_obstacles.trajectory_evaluation = eval_results

    return collision_xy, mpc, traj_agent_idx, trajectory_full, scenario_obstacles


def create_following_trajectory(state, trajectories_full):
    # Add one trajectory if the ego keeps stay on its current lane following the cyclist
    # Pick one of the trajectories_full in order calculate completion_time
    resampled_trajectory = compute_predicted_trajectory(state, trajectories_full[0][0])
    completion_time = calculate_trajectory_completion_time(resampled_trajectory, state)
    # Get initial values
    init_x = resampled_trajectory[0, 0]  # Initial x value
    init_y = resampled_trajectory[0, 1]  # Initial y value
    init_theta = resampled_trajectory[0, 2]  # Initial theta value
    original_length = len(resampled_trajectory)
    # Create a copy of the trajectory to modify
    follow_trajectory = resampled_trajectory.copy()
    # Create an array of y-positions with proper length
    new_y_values = np.arange(
        init_y,
        init_y + (completion_time * state.v),
        (state.v * ScenarioParameters.DT)
    )
    # Ensure the array has the correct length
    if len(new_y_values) != original_length:
        # If too short, extend the array by repeating the last value
        if len(new_y_values) < original_length:
            new_y_values = np.append(
                new_y_values,
                np.repeat(new_y_values[-1], original_length - len(new_y_values))
            )
        # If too long, truncate the array
        else:
            new_y_values = new_y_values[:original_length]
    # Now assign the correctly sized array for y-coordinate (column 1)
    follow_trajectory[:, 1] = new_y_values
    # Keep x-coordinate (column 0) constant at the initial value
    follow_trajectory[:, 0] = init_x
    # Keep orientation (column 2) constant at the initial value
    follow_trajectory[:, 2] = init_theta
    return follow_trajectory


def visualize_trajectory_evaluations(eval_results, trajectories_full, moving_obstacles, state, car_dimensions,
                                     bicycle_dimensions, scenario, time_values,
                                     reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values,
                                     agent_weights,
                                     save_path=None):
    """
    Visualize trajectory evaluations with two separate outputs:
    1. Individual trajectory plots showing dynamic reasons
    2. Spatial plot showing car, obstacles, and trajectories

    Args:
        eval_results: Evaluation results from evaluate_trajectories_for_reasons
        trajectories_full: List of trajectories
        moving_obstacles: List of moving obstacles
        state: Current vehicle state
        car_dimensions: Vehicle dimensions
        bicycle_dimensions: Bicycle dimensions
        scenario: Scenario object containing environment information
        save_path: Optional path to save the visualization
    """
    if not trajectories_full:
        logger.warning("No trajectories to visualize")
        return

    # Define trajectory types for subtitles
    trajectory_descriptions = {
        0: "Small-Gap Overtaking",
        1: "Medium-Gap Overtaking",
        2: "Large-Gap Overtaking",
        3: "Conservative Following"
    }

    # Only use up to 4 trajectories
    max_trajectories = min(4, len(trajectories_full))
    trajectories_full = trajectories_full[:max_trajectories]

    # Colors for different trajectories with better visibility
    colors = ['#0000FF', '#000000', '#FF0000', '#BF00C0']
    line_styles = ['--', '-.', '-', '--']  # Cycle through styles
    line_widths = [2.0, 2.0, 2.0, 2.0]  # Consistent widths

    # Create trajectory styling with trajectory info and line properties
    trajectory_styles = []
    for i, trajectory in enumerate(trajectories_full):
        # For trajectories with tuple structure (from run_all)
        if isinstance(trajectory, tuple):
            trajectory_points = trajectory[0]
        else:
            trajectory_points = trajectory

        # Extract evaluation data
        eval_data = eval_results['all_evaluations'][i]
        score = eval_data['total_score']

        # Store trajectory info along with styling
        trajectory_styles.append({
            'trajectory': trajectory_points,
            'color': colors[i % len(colors)],
            'style': line_styles[i % len(line_styles)],
            'width': line_widths[i % len(line_widths)],
            'score': score,
            'index': i,
            'eval_data': eval_data
        })

    # FIRST OUTPUT: Create trajectory plots (2x2 grid)
    fig_trajectories = plt.figure(figsize=(20, 14))
    gs_trajectories = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Create trajectory plot subplots
    traj_axes = []
    for i in range(max_trajectories):
        row = i // 2  # 0, 0, 1, 1
        col = i % 2  # 0, 1, 0, 1
        ax = plt.subplot(gs_trajectories[row, col])
        traj_axes.append(ax)

    # Plot the individual trajectory plots with dynamic reasons
    for i, style_info in enumerate(trajectory_styles):
        if i >= max_trajectories:
            break

        ax = traj_axes[i]
        color = style_info['color']
        style = style_info['style']
        width = style_info['width'] * 2  # Double the line width
        i = style_info['index']
        eval_data = style_info['eval_data']
        completion_time = eval_data['completion_time']

        # Set main title for each trajectory plot with clarified score meaning (higher position)
        ax.set_title(f"Trajectory {i + 1}: Total Score $S(T_a)$ = {style_info['score']:.3f}",
                     fontsize=28, pad=40)  # Normal padding for title

        # Add trajectory description BELOW the title
        traj_description = trajectory_descriptions.get(i, f"Trajectory Type {i}")
        ax.text(0.5, 1.02, traj_description,
                transform=ax.transAxes,
                ha='center', fontsize=28,
                color='#555555')

        # Adjust the subplot to create more space at the top
        plt.subplots_adjust(top=0.85)  # Increase this value to create more space

        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Time [s]", fontsize=24)
        ax.set_ylabel("Reasons [0-1]", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.15)  # Very light grid

        # Plot the individual reason scores for each trajectory
        reason_types = [
            ("Policymaker", 'policymaker', eval_data['detailed_scores']['policymaker'], '#0072B2'),  # Blue
            ("Driver", 'driver', eval_data['detailed_scores']['driver'], '#D55E00'),  # Orange
            ("Cyclist", 'cyclist_combined', eval_data['detailed_scores']['cyclist_combined'], '#009E73')  # Green
        ]

        # Track if we have historical data
        has_historical = isinstance(time_values, list) and len(time_values) > 0

        # If historical data exists, plot it with light background
        if has_historical:
            historical_start = time_values[0] if np.isscalar(time_values[0]) else time_values[0]
            historical_end = time_values[-1] if np.isscalar(time_values[-1]) else time_values[-1]

            # Add a colored background rectangle for historical data
            historical_rect = plt.Rectangle(
                (historical_start, 0),
                width=(historical_end - historical_start),
                height=1.1,
                color='#F7E2DA',
                alpha=0.5,
                zorder=0
            )
            ax.add_patch(historical_rect)

            # Add a vertical line at the transition point
            ax.axvline(x=historical_end, color='#D3B1A6', linestyle='--', linewidth=2, alpha=0.7)

            # Plot historical values for all reason types
            reason_values_map = {
                'policymaker': reasons_policymaker_values,
                'driver': reasons_driver_values,
                'cyclist_combined': reasons_cyclist_values
            }

            # Plot historical data for each reason (all in black)
            for _, reason_key, _, _ in reason_types:
                ax.plot(
                    time_values,
                    reason_values_map[reason_key],
                    color='black',
                    linestyle='-',
                    linewidth=3,  # Thicker historical lines
                    alpha=0.7
                )

        # Generate time points for each trajectory's forecast
        for reason_name, reason_key, scores, reason_color in reason_types:
            if len(scores) > 0:
                # Calculate time points for this trajectory
                if has_historical:
                    # For the first call, time_values[-1] might be a number
                    last_time = time_values[-1] if np.isscalar(time_values[-1]) else time_values[-1]
                    start_time = last_time + ScenarioParameters.DT
                else:
                    start_time = 0

                end_time = start_time + completion_time
                num_full_steps = int((end_time - start_time) / 0.1)  # Number of complete 0.1s intervals
                remaining_time = (end_time - start_time) - (num_full_steps * 0.1)  # Any remaining time

                # Generate the regular 0.1s intervals
                regular_points = np.array([start_time + i * 0.1 for i in range(num_full_steps)])

                # Add the final point if there's a remainder
                if remaining_time > 0:
                    time_points = np.append(regular_points, end_time)
                else:
                    time_points = regular_points

                # Ensure we have the right number of points
                if len(time_points) < len(scores):
                    # If we need more points, add intermediate points
                    missing_points = len(scores) - len(time_points)
                    additional_points = np.linspace(start_time, end_time, missing_points + 2)[1:-1]
                    time_points = np.sort(np.concatenate([time_points, additional_points]))
                elif len(time_points) > len(scores):
                    # If we have too many points, truncate
                    time_points = time_points[:len(scores)]

                # Add a colored background rectangle for the scoring region
                if reason_name == "Policymaker":  # Only add once per trajectory
                    scoring_rect = plt.Rectangle(
                        (time_points[0], 0),
                        width=(time_points[-1] - time_points[0]),
                        height=1.1,
                        color='#E6F5FF',  # Light blue
                        alpha=0.3,
                        zorder=1
                    )
                    ax.add_patch(scoring_rect)

                # Plot each reason type with its own color
                ax.plot(
                    time_points,
                    scores,
                    color=reason_color,
                    linestyle='-' if reason_name == "Policymaker" else '--' if reason_name == "Driver" else ':',
                    linewidth=4,  # Thicker lines for better visibility
                    label=reason_name
                )

        # Add legend to each plot
        # ax.legend(loc='center left', fontsize=20)

        # Group the annotations for both regions together at the bottom
        # This creates a cleaner design and makes comparison easier

        # Add annotation for historical data at the bottom left
        if has_historical:
            ax.text(
                historical_start + 0.1,  # Slight offset from left edge
                0.10,  # Position near bottom
                'Historical Data\n(Not Used for $S(T_a)$)',
                ha='left',
                va='center',
                fontsize=20,
                color='#8B0000',  # Dark red
                bbox=dict(facecolor='#F7E2DA', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='#D3B1A6')
            )

        # Add annotation for scoring region at the bottom right
        if len(scores) > 0:
            ax.text(
                time_points[-1] - 0.1,  # Slight offset from right edge
                0.10,  # Position near bottom
                'Region Used for\n$S(T_a)$ Calculation',
                ha='right',
                va='center',
                fontsize=20,
                color='#00008B',  # Dark blue
                bbox=dict(facecolor='#E6F5FF', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='#99CCFF')
            )

    # Apply tight layout to trajectory figure with more top space
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
    plt.subplots_adjust(top=0.93)  # More space at top

    # Save trajectory plots if a save path is provided
    if save_path:
        trajectories_save_path = save_path.replace('.png', '_trajectories.png')
        plt.savefig(trajectories_save_path, dpi=400, bbox_inches='tight')

    # Show the trajectory plots
    plt.show()

    # SECOND OUTPUT: Create spatial plot with fixed dimensions
    # Use a wider, larger figure
    fig_spatial = plt.figure(figsize=(16, 12))  # Large figure

    # Create axes that use the entire figure area with no margins
    ax_spatial = fig_spatial.add_axes([0.01, 0.05, 0.98, 0.85])  # Nearly full figure

    # Set background color for spatial plot
    ax_spatial.set_facecolor('#AFABAB')

    # Plot car at current state
    car_polygon = draw_car((state.x, state.y, state.yaw), car_dimensions, ax=ax_spatial, color='black')

    # Plot bicycle/moving obstacles
    for i, obstacle in enumerate(moving_obstacles):
        obstacle_x, obstacle_y, _, obstacle_yaw, _, _ = obstacle.get()
        draw_bicycle((obstacle_x, obstacle_y, obstacle_yaw), bicycle_dimensions, ax=ax_spatial, color='blue')

    # Draw all static elements from scenario
    draw_static_elements(ax_spatial, scenario)

    # Plot trajectories in the spatial plot
    for style_info in trajectory_styles:
        trajectory_points = style_info['trajectory']
        color = style_info['color']
        style = style_info['style']
        width = style_info['width']
        score = style_info['score']
        i = style_info['index']

        # Get trajectory description
        # traj_description = trajectory_descriptions.get(i, f"Trajectory Type {i}")

        # Plot main trajectory line with improved label
        ax_spatial.plot(
            trajectory_points[:, 0], trajectory_points[:, 1],
            color=color,
            linestyle=style,
            linewidth=width * 2,  # Make lines thicker for better visibility
            # label=f"Traj {i}: {traj_description} (Score: {score:.3f})"  # Enhanced label
            label=f"Trajectory {i + 1}"  # Enhanced label
        )

    goal_x, goal_y = ScenarioParameters.X_LOC_GOAL, ScenarioParameters.Y_LOC_GOAL
    ax_spatial.add_patch(plt.Rectangle(
        (goal_x - 1, goal_y - 0.5),  # Bottom-left corner of the rectangle
        2, 1,  # Width and height of the rectangle
        color='orange', alpha=0.5, zorder=2  # Transparent orange
    ))
    ax_spatial.text(
        goal_x, goal_y, "GOAL", color='black', fontsize=12, ha='center', va='center', zorder=3
    )

    # Add title to the spatial plot with more padding
    # priority_label = get_priority_label(agent_weights)
    # weights_label = format_weights(agent_weights)
    # ax_spatial.set_title(f"{priority_label} \n {weights_label}",
    #                      fontsize=16, pad=20)  # Increased padding

    # CRITICAL CHANGE: Force specific axis limits
    car_x, car_y = state.x, state.y
    # Set very specific limits that focus tightly on the car and immediate surroundings
    # These values should be adjusted based on your specific scenario dimensions
    ax_spatial.set_xlim(car_x - 12, car_x + 10)  # 14 units wide centered on car
    ax_spatial.set_ylim(car_y - 2, ScenarioParameters.Y_LOC_GOAL + 2)  # More space ahead than behind

    # Set aspect ratio to equal - IMPORTANT for consistent scaling
    ax_spatial.set_aspect('equal')

    # Keep minimal grid lines for reference
    ax_spatial.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove all tick labels but keep the ticks for reference
    # ax_spatial.set_xticklabels([])
    # ax_spatial.set_yticklabels([])
    ax_spatial.tick_params(axis='both', which='major', labelsize=14)  # Set label size to 14

    # Trajectory legend - moved to upper left to avoid covering trajectories
    handles, labels = ax_spatial.get_legend_handles_labels()
    traj_nums = [int(label.split(':')[0].split(' ')[1]) for label in labels]
    sorted_pairs = sorted(zip(handles, labels, traj_nums), key=lambda x: x[2])
    legend = ax_spatial.legend([pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs],
                               loc='upper left', fontsize=12, framealpha=0.7)
    legend.get_frame().set_linewidth(0.5)

    # Save spatial plot if a save path is provided
    if save_path:
        spatial_save_path = save_path.replace('.png', '_spatial.png')
        plt.savefig(spatial_save_path, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # Show the spatial plot
    plt.show()

    return fig_trajectories, fig_spatial


# def visualize_trajectory_evaluations(eval_results, trajectories_full, moving_obstacles, state, car_dimensions,
#                                      bicycle_dimensions, scenario, time_values,
#                                      reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, agent_weights,
#                                      save_path=None):
#     """
#     Visualize trajectory evaluations to compare different options using a 3×2 layout.
#
#     Args:
#         eval_results: Evaluation results from evaluate_trajectories_for_reasons
#         trajectories_full: List of trajectories
#         moving_obstacles: List of moving obstacles
#         state: Current vehicle state
#         car_dimensions: Vehicle dimensions
#         bicycle_dimensions: Bicycle dimensions
#         scenario: Scenario object containing environment information
#         save_path: Optional path to save the visualization
#     """
#     if not trajectories_full:
#         logger.warning("No trajectories to visualize")
#         return
#
#     # Create figure with GridSpec for 3×2 layout
#     fig = plt.figure(figsize=(10, 10))
#     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.5, 1])
#
#     # Create subplots - trajectory plot spans all rows in column 1
#     ax_spatial = plt.subplot(gs[:, 0])  # Left column: Trajectories (spans all rows)
#     ax_policy = plt.subplot(gs[0, 1])  # Top-right: Policy scores
#     ax_driver = plt.subplot(gs[1, 1])  # Middle-right: Driver scores
#     ax_cyclist = plt.subplot(gs[2, 1])  # Bottom-right: Cyclist scores
#
#     # Set background color for spatial plot
#     ax_spatial.set_facecolor('#AFABAB')
#
#     # Colors for different trajectories with better visibility
#     colors = ['#0000FF', '#000000', '#FF0000', '#BF00C0', '#00BFBF', '#8c564b']
#     line_styles = ['--', '-.', '-.', '--', '-.', '--']  # Cycle through styles
#     line_widths = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # Varying widths
#
#     # Create trajectory styling with trajectory info and line properties
#     trajectory_styles = []
#     for i, trajectory in enumerate(trajectories_full):
#         # For trajectories with tuple structure (from run_all)
#         if isinstance(trajectory, tuple):
#             trajectory_points = trajectory[0]
#         else:
#             trajectory_points = trajectory
#
#         # Extract evaluation data
#         eval_data = eval_results['all_evaluations'][i]
#         score = eval_data['total_score']
#
#         # Store trajectory info along with styling
#         trajectory_styles.append({
#             'trajectory': trajectory_points,
#             'color': colors[i % len(colors)],
#             'style': line_styles[i % len(line_styles)],
#             'width': line_widths[i % len(line_widths)],
#             'score': score,
#             'index': i,
#             'eval_data': eval_data
#         })
#
#     # Sort trajectory styles by line width (thickest first)
#     trajectory_styles.sort(key=lambda x: x['width'], reverse=True)
#
#     # Plot car at current state
#     car_polygon = draw_car((state.x, state.y, state.yaw), car_dimensions, ax=ax_spatial, color='black')
#
#     # Plot bicycle/moving obstacles
#     for i, obstacle in enumerate(moving_obstacles):
#         obstacle_x, obstacle_y, _, obstacle_yaw, _, _ = obstacle.get()
#         draw_bicycle((obstacle_x, obstacle_y, obstacle_yaw), bicycle_dimensions, ax=ax_spatial, color='blue')
#
#     # Draw all static elements from scenario
#     draw_static_elements(ax_spatial, scenario)
#
#     # Plot trajectories in order of line width (thickest first, thinnest last)
#     for style_info in trajectory_styles:
#         trajectory_points = style_info['trajectory']
#         color = style_info['color']
#         style = style_info['style']
#         width = style_info['width']
#         score = style_info['score']
#         i = style_info['index']
#
#         # Plot main trajectory line with simplified label
#         ax_spatial.plot(
#             trajectory_points[:, 0], trajectory_points[:, 1],
#             color=color,
#             linestyle=style,
#             linewidth=width,
#             label=f"Traj {i}: {score:.3f}"  # Simplified label
#         )
#
#         # Add title to the spatial plot
#         priority_label = get_priority_label(agent_weights)
#         weights_label = format_weights(agent_weights)
#         ax_spatial.set_title(f"{priority_label} \n {weights_label}", fontsize=16, pad=10)
#
#         # Set plot limits and aspect ratio
#         ax_spatial.set_aspect('equal')
#         ax_spatial.grid(True, alpha=0.3)  # Keep grid only for spatial plot
#
#         # Add arrow to show direction
#         mid_idx = len(trajectory_points) // 2
#         if mid_idx > 0:
#             arrow_start = trajectory_points[mid_idx - 1, :2]
#             arrow_end = trajectory_points[mid_idx, :2]
#             dx = arrow_end[0] - arrow_start[0]
#             dy = arrow_end[1] - arrow_start[1]
#             ax_spatial.arrow(
#                 arrow_start[0], arrow_start[1], dx, dy,
#                 head_width=0.3, head_length=0.6, fc=color, ec=color
#             )
#
#     # Set plot limits and aspect ratio
#     ax_spatial.set_aspect('equal')
#     ax_spatial.grid(True, alpha=0.3)  # Keep grid only for spatial plot
#
#     # Set appropriate axis limits
#     car_x, car_y = state.x, state.y
#     ax_spatial.set_xlim(car_x - 20, car_x + 20)
#     ax_spatial.set_ylim(car_y - 5, car_y + 52)
#
#     # Trajectory legend inside the plot (to save space)
#     # Sort legend entries by trajectory index for consistency
#     handles, labels = ax_spatial.get_legend_handles_labels()
#     # Extract trajectory numbers and sort by them
#     traj_nums = [int(label.split(':')[0].split(' ')[1]) for label in labels]
#     sorted_pairs = sorted(zip(handles, labels, traj_nums), key=lambda x: x[2])
#     ax_spatial.legend([pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs],
#                       loc='upper right', fontsize=10)
#
#     # Plot reason scores
#     reason_plots = [
#         (ax_policy, "Policymaker Reasons", 'policymaker'),
#         (ax_driver, "Driver Reasons", 'driver'),
#         (ax_cyclist, "Cyclist Reasons", 'cyclist_combined')
#     ]
#
#     # Plot each reason with consistent styling, but in order of line width
#     for ax, title, score_key in reason_plots:
#         ax.set_title(title, fontsize=16)
#         ax.set_ylim(0, 1.1)
#         ax.set_xlabel("Time [s]", fontsize=14)
#         ax.set_ylabel("Reasons [0-1]", fontsize=14)
#         ax.tick_params(axis='both', which='major', labelsize=12)
#
#         # Remove grid (change #2)
#         ax.grid(False)
#
#         # Keep white background (change #2)
#         # No gray background - removed the line: ax.axhspan(0.0, 1.0, alpha=0.1, color='gray')
#
#         # Create style info for reason lines, but sorted by width
#         reason_styles = []
#         # Track minimum scores for this reason type
#         min_score = 1.0  # Start with maximum possible (1.0)
#
#         for style_info in trajectory_styles:
#             i = style_info['index']
#             scores = style_info['eval_data']['detailed_scores'][score_key]
#             # Update minimum score if any value is lower
#             if len(scores) > 0:
#                 min_score = min(min_score, np.min(scores))
#
#             reason_styles.append({
#                 'scores': scores,
#                 'color': style_info['color'],
#                 'style': style_info['style'],
#                 'width': style_info['width'],
#                 'index': i,
#                 'completion_time': style_info['eval_data']['completion_time']
#             })
#
#         # Map of score keys to reason values lists
#         reason_values_map = {
#             'policymaker': reasons_policymaker_values,
#             'driver': reasons_driver_values,
#             'cyclist_combined': reasons_cyclist_values
#         }
#
#         # Check historical data for minimum as well
#         if isinstance(reason_values_map[score_key], list) and len(reason_values_map[score_key]) > 0:
#             min_score = min(min_score, min(reason_values_map[score_key]))
#
#         # Set y-axis limits based on minimum score with some padding
#         # Round down to nearest 0.1 to have clean axis limits
#         min_y = max(0, math.floor(min_score * 10) / 10)  # Never go below 0
#         ax.set_ylim(min_y - 0.1, 1.1)
#         # Sort by line width (thickest first)
#         reason_styles.sort(key=lambda x: x['width'], reverse=True)
#
#         # Plot each trajectory's scores with consistent styling
#         for style_info in reason_styles:
#             color = style_info['color']
#             style = style_info['style']
#             width = style_info['width']
#             scores = style_info['scores']
#             i = style_info['index']
#             completion_time = style_info['completion_time']
#
#             time_values_ = copy.deepcopy(time_values)
#             reasons_policymaker_values_ = copy.deepcopy(reasons_policymaker_values)
#             reasons_driver_values_ = copy.deepcopy(reasons_driver_values)
#             reasons_cyclist_values_ = copy.deepcopy(reasons_cyclist_values)
#
#             # Map of score keys to reason values lists
#             reason_values_map = {
#                 'policymaker': reasons_policymaker_values_,
#                 'driver': reasons_driver_values_,
#                 'cyclist_combined': reasons_cyclist_values_
#             }
#
#             # Calculate the range of historical data (before extension)
#             if isinstance(time_values, list) and len(time_values) > 0:
#                 historical_start = time_values[0] if np.isscalar(time_values[0]) else time_values[0]
#                 historical_end = time_values[-1] if np.isscalar(time_values[-1]) else time_values[-1]
#
#                 # Add a colored background rectangle for historical data
#                 historical_rect = plt.Rectangle(
#                     (historical_start, 0),
#                     width=(historical_end - historical_start),
#                     height=1.1,
#                     color='#F7E2DA',
#                     alpha=0.5,
#                     zorder=0  # Ensure it's behind the plotted lines
#                 )
#                 ax.add_patch(historical_rect)
#
#                 # Optionally add a vertical line at the transition point
#                 ax.axvline(x=historical_end, color='#D3B1A6', linestyle='--', linewidth=1, alpha=0.7)
#
#                 # Add a label
#                 ax.text(
#                     (historical_start + historical_end)/2,
#                     1.05,
#                     'Before replanner triggered',
#                     ha='center',
#                     va='center',
#                     fontsize=8,
#                     bbox=dict(facecolor='#F7E2DA', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='#D3B1A6')
#                 )
#
#                 # Plot the scores - accessing the specific list for this score_key
#                 ax.plot(
#                     time_values_, reason_values_map[score_key],  # Access the list for this specific key
#                     color='black')
#
#             # If you need to keep track of all time points
#             if isinstance(time_values_, list):
#                 # For the first call, time_values[-1] might be a number
#                 last_time = time_values_[-1] if np.isscalar(time_values_[-1]) else time_values_[-1]
#
#                 # Generate new time points with fixed 0.1s intervals
#                 start_time = last_time + ScenarioParameters.DT
#                 end_time = start_time + completion_time
#                 num_full_steps = int((end_time - start_time) / 0.1)  # Number of complete 0.1s intervals
#                 remaining_time = (end_time - start_time) - (num_full_steps * 0.1)  # Any remaining time
#
#                 # Generate the regular 0.1s intervals
#                 regular_points = np.array([start_time + i * 0.1 for i in range(num_full_steps)])
#
#                 # Add the final point if there's a remainder
#                 if remaining_time > 0:
#                     time_points = np.append(regular_points, end_time)
#                 else:
#                     time_points = regular_points
#
#                 # Ensure we have the right number of points
#                 if len(time_points) < len(scores):
#                     # If we need more points, add intermediate points (this shouldn't normally happen)
#                     missing_points = len(scores) - len(time_points)
#                     additional_points = np.linspace(start_time, end_time, missing_points + 2)[1:-1]
#                     time_points = np.sort(np.concatenate([time_points, additional_points]))
#                 elif len(time_points) > len(scores):
#                     # If we have too many points, truncate
#                     time_points = time_points[:len(scores)]
#
#             #     # Use extend to append individual values (not the whole array)
#             #     time_values_.extend(time_points)
#             #
#             #     # Use these new time points for plotting the current trajectory
#             #     plot_times = time_points
#             # else:
#             #     # If time_values is not a list, create new time points from 0
#             #     time_points = np.linspace(0, completion_time, len(scores))
#             #     plot_times = time_points
#             #
#             # # Extend the appropriate reason values list
#             # if isinstance(reason_values_map[score_key], list):
#             #     reason_values_map[score_key].extend(scores)  # Extend with individual values
#             # else:
#             #     reason_values_map[score_key] = list(scores)  # Convert to list if not already
#
#             # Plot the scores - accessing the specific list for this score_key
#             ax.plot(
#                 time_points, scores,  # Access the list for this specific key
#                 color=color, linestyle=style, linewidth=width
#             )
#
#
#         # Remove safety threshold line (change #1)
#         # Removed the code: ax.axhline(y=0.3, color='r', linestyle=':', linewidth=2, label='Safety Threshold')
#
#     # Add a common legend to the first plot only (top-right)
#     # Sort legend entries by trajectory index for consistency
#     handles, labels = ax_policy.get_legend_handles_labels()
#
#     # Sort by trajectory number
#     traj_labels = [l for l in labels if l.startswith('Traj')]
#     if traj_labels:
#         # Extract trajectory numbers and sort by them
#         traj_nums = [int(label.split(' ')[1]) for label in traj_labels]
#         sorted_idx = sorted(range(len(traj_nums)), key=lambda i: traj_nums[i])
#
#         sorted_handles = [handles[i] for i in sorted_idx]
#         sorted_labels = [labels[i] for i in sorted_idx]
#
#         ax_policy.legend(sorted_handles, sorted_labels,
#                          loc='upper center', ncol=1, fontsize=9, bbox_to_anchor=(0.2, 0.6))
#
#     # Remove redundant legends from other plots
#     if ax_driver.get_legend():
#         ax_driver.get_legend().remove()
#     if ax_cyclist.get_legend():
#         ax_cyclist.get_legend().remove()
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=800, bbox_inches='tight')
#
#     plt.show()

def get_priority_label(agent_weights):
    """
    Return a human-readable label based on the agent weights.

    Args:
        agent_weights: Dictionary containing weights for different agents (policymaker, driver, cyclist)

    Returns:
        str: A descriptive label indicating the priority focus
    """
    # Extract weights from the dictionary
    policy_weight = agent_weights.get('policymaker', 0.0)
    driver_weight = agent_weights.get('driver', 0.0)
    cyclist_weight = agent_weights.get('cyclist', 0.0)

    # Determine the priority based on the highest weight
    max_weight = max(policy_weight, driver_weight, cyclist_weight)

    # Small threshold for considering weights approximately equal
    threshold = 0.05

    if max_weight == driver_weight and driver_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Driver Reasons Weighted Higher)"
    elif max_weight == policy_weight and policy_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Policymaker Reasons Weighted Higher)"
    elif max_weight == cyclist_weight and cyclist_weight >= 0.5:
        return "Trajectory Scoring Based on Human Reasons \n (Cyclist Reasons Weighted Higher)"
    elif abs(driver_weight - policy_weight) < threshold and abs(driver_weight - cyclist_weight) < threshold:
        return "Trajectory Scoring Based on Human Reasons \n (All Reasons Weighted The Same)"
    else:
        return "Mixed Priority"


def format_weights(agent_weights):
    """
    Format agent weights as a concise string.

    Args:
        agent_weights: Dictionary containing weights for different agents (policymaker, driver, cyclist)

    Returns:
        str: A formatted string representation of the weights
    """
    # Extract weights from the dictionary
    policy_weight = agent_weights.get('policymaker', 0.0)
    driver_weight = agent_weights.get('driver', 0.0)
    cyclist_weight = agent_weights.get('cyclist', 0.0)

    # Format with D for driver, P for policy, C for cyclist
    if policy_weight == 1 / 3 or driver_weight == 1 / 3 or cyclist_weight == 1 / 3:
        return "Driver:1/3; Policymaker:1/3; Cyclist:1/3"
    else:
        return f"Driver:{driver_weight:.3f}; Policymaker:{policy_weight:.3f}; Cyclist:{cyclist_weight:.3f}"


def balance_function(weights, ideal_weights=None):
    """
    Calculate the balance function value based on deviation from ideal weights
    and ensuring minimum representation.

    Parameters:
    - weights: List of weight values that sum to 1
    - ideal_weights: Optional custom ideal weight distribution. If None, equal weights (1/n) are used

    Returns:
    - 1.0 for weights matching the ideal distribution
    - 0.0 when any stakeholder has zero weight
    - Values between 0-1 based on deviation from ideal and minimum representation
    """
    n = len(weights)

    # Use custom ideal weights if provided, otherwise default to equal weights
    if ideal_weights is None:
        ideal_weights = [1 / n] * n

    # Ensure ideal_weights is the same length as weights
    if len(ideal_weights) != n:
        raise ValueError("ideal_weights must have the same length as weights")

    # Calculate minimum weight ratio relative to its ideal weight
    # This ensures the score is zero when any weight is zero
    min_weight_ratio = min(w / i for w, i in zip(weights, ideal_weights))

    # Calculate distribution balance using RMS deviation from ideal weights
    sum_squared_deviations = sum((w - i) ** 2 for w, i in zip(weights, ideal_weights))
    rms_deviation = np.sqrt(sum_squared_deviations / n)

    # Normalize the deviation (maximum possible deviation in an n-dimensional space)
    # Maximum deviation can be calculated as the distance between the ideal point and the furthest corner
    max_deviation = np.sqrt(sum(i ** 2 for i in ideal_weights))  # This is a simplification and may need refinement

    # Normalize to get a value between 0 and 1
    distribution_balance = 1 - (rms_deviation / max_deviation)

    # Combine the two balance measures
    return distribution_balance * min_weight_ratio


def evaluate_trajectories_for_reasons(weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist,
                                      ideal_weight_driver, ideal_weight_policymaker, trajectories_full,
                                      moving_obstacles, state, car_dimensions, bicycle_dimensions,
                                      reasons_cyclist_comfort, reasons_driver_time_eff,
                                      reasons_policymaker_reg_compliance,
                                      time_elapsed_driver=0.0, time_passed_cyclist=0.0):
    """
    Evaluate multiple trajectories based on human-centered reasons.

    Args:
        trajectories_full: List of candidate trajectories
        moving_obstacles: List of moving obstacles (bicycle)
        state: Current state of the vehicle
        car_dimensions: Dimensions of the car
        bicycle_dimensions: Dimensions of the bicycle

    Returns:
        dict: Evaluation results with scores and best trajectory
    """
    trajectory_scores = []
    detailed_evaluations = []

    # For each trajectory option
    for i, trajectory in enumerate(trajectories_full):
        # 1. Calculate time needed to complete this trajectory
        # For the last trajectory, we need to calculate resampled_trajectory differently
        if i == len(trajectories_full) - 1:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0], last_index=True)
            # set completion time as the real completion time from other trajectories
            # completion_time = calculate_trajectory_completion_time(resampled_trajectory, state, last_index=True)
        else:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0])
            completion_time = calculate_trajectory_completion_time(resampled_trajectory, state)

        # 2. Predict bicycle movement for this duration
        bicycle_future_trajectory = np.vstack(MovingObstaclesPrediction(
            *moving_obstacles[0].get(),
            sample_time=ScenarioParameters.DT,
            car_dimensions=bicycle_dimensions
        ).state_prediction(completion_time)).T

        # 3. Sample points along both trajectories
        num_sample_points = len(resampled_trajectory)

        # Create evenly spaced indices
        ego_indices = np.linspace(0, len(resampled_trajectory) - 1, num_sample_points, dtype=int)
        # Delete the last point of bicycle_future_trajectory
        bicycle_indices = np.linspace(0, len(bicycle_future_trajectory[:-1]) - 1, num_sample_points, dtype=int)

        # 4. Evaluate reasons at each sample point
        current_time_elapsed_driver = time_elapsed_driver
        current_time_passed_cyclist = time_passed_cyclist

        policymaker_scores = []
        driver_scores = []
        cyclist_comfort_scores = []
        cyclist_time_scores = []
        cyclist_combined_scores = []

        # Width of the car for centerline evaluation
        car_width = car_dimensions.bounding_box_size[0]

        for j in range(num_sample_points):
            # Get ego vehicle state at this point
            ego_x = resampled_trajectory[ego_indices[j], 0]
            ego_y = resampled_trajectory[ego_indices[j], 1]
            ego_theta = resampled_trajectory[ego_indices[j], 2]

            # Get bicycle state at this point
            bicycle_x = bicycle_future_trajectory[bicycle_indices[j], 0]
            bicycle_y = bicycle_future_trajectory[bicycle_indices[j], 1]

            # Create a simulated state object for evaluation
            simulated_state = State(x=ego_x, y=ego_y, yaw=ego_theta, v=state.v)

            # Create a simulated obstacle object for evaluation
            simulated_obstacle = type('obj', (object,), {
                'get': lambda self=None: (bicycle_x, bicycle_y, 0, 0, CyclistParameters.SPEED, 0)
            })
            simulated_obstacles = [simulated_obstacle]

            # Evaluate policymaker (regulatory compliance)
            policymaker_score = evaluate_distance_to_centerline(
                ego_x, car_width, ScenarioParameters.CENTERLINE_LOCATION)
            policymaker_scores.append(policymaker_score)

            # Evaluate driver time efficiency
            driver_score, current_time_elapsed_driver = evaluate_time_following(
                'driver_reasons', ScenarioParameters.DT,
                DriverParameters.DISTANCE_BUFFER, DriverParameters.DISTANCE_REF,
                DriverParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_elapsed_driver)
            driver_scores.append(driver_score)

            # Evaluate cyclist comfort (distance)
            cyclist_comfort_score = evaluate_distance_to_obstacle(
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                simulated_obstacles, simulated_state)
            cyclist_comfort_scores.append(cyclist_comfort_score)

            # Evaluate cyclist comfort (time)
            cyclist_time_score, current_time_passed_cyclist = evaluate_time_following(
                'cyclist_reasons', ScenarioParameters.DT,
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                CyclistParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_passed_cyclist)
            cyclist_time_scores.append(cyclist_time_score)

            # Combined cyclist score
            cyclist_combined_score = cyclist_comfort_score * cyclist_time_score
            cyclist_combined_scores.append(cyclist_combined_score)

        # delete last value of scores because the ego vehicle is not moving but the bicycle is
        policymaker_scores = policymaker_scores[:-1]
        driver_scores = driver_scores[:-1]
        cyclist_combined_scores = cyclist_combined_scores[:-1]

        # replace first value of reasons scores with the first value of reasons evaluation
        policymaker_scores[0] = reasons_policymaker_reg_compliance
        driver_scores[0] = reasons_driver_time_eff
        cyclist_combined_scores[0] = reasons_cyclist_comfort

        # 5. Calculate average scores for each reason
        avg_policymaker = np.mean(policymaker_scores[:-1])
        avg_driver = np.mean(driver_scores)
        avg_cyclist = np.mean(cyclist_combined_scores)

        # 6. Calculate weighted total score
        # Define weights for different agents (matching your existing weights)
        agent_weights = {
            'policymaker': weightofpolicy,  # Regulatory compliance
            'driver': weightofdriver,  # Driver patience/efficiency
            'cyclist': weightofcyclist  # Cyclist comfort/safety
        }

        print(f"Agent Weights: {agent_weights}")

        # Calculate balance function value
        ideal_weights = [ideal_weight_cyclist, ideal_weight_driver,
                         ideal_weight_policymaker]  # Cyclist, Driver, Policymaker
        # ideal_weights = [0.1, 0.1, 0.8]  # Cyclist, Driver, Policymaker -> more focus on policymaker
        balance_value = balance_function(
            [agent_weights.get('cyclist', 0), agent_weights.get('driver', 0), agent_weights.get('policymaker', 0)],
            ideal_weights=ideal_weights)

        total_score = balance_value * (
                agent_weights['policymaker'] * avg_policymaker +
                agent_weights['driver'] * avg_driver +
                agent_weights['cyclist'] * avg_cyclist
        )

        # Store results
        trajectory_scores.append(total_score)

        detailed_eval = {
            'trajectory_idx': i,
            'total_score': total_score,
            'completion_time': completion_time,
            'avg_scores': {
                'policymaker': avg_policymaker,
                'driver': avg_driver,
                'cyclist': avg_cyclist
            },
            'detailed_scores': {
                'policymaker': policymaker_scores,
                'driver': driver_scores,
                'cyclist_comfort': cyclist_comfort_scores,
                'cyclist_time': cyclist_time_scores,
                'cyclist_combined': cyclist_combined_scores
            }
        }

        detailed_evaluations.append(detailed_eval)

        logger.info(f"Trajectory {i}: Score {total_score:.3f} (P: {avg_policymaker:.2f}, "
                    f"D: {avg_driver:.3f}, C: {avg_cyclist:.3f}) "
                    f"Time: {completion_time:.3f}s")

    # 7. Find the best trajectory
    if trajectory_scores:
        best_idx = np.argmax(trajectory_scores)
        best_score = trajectory_scores[best_idx]
        best_trajectory = trajectories_full[best_idx]
        best_evaluation = detailed_evaluations[best_idx]

        logger.info(f"Selected trajectory {best_idx} with score {best_score:.3f}")
    else:
        best_idx = None
        best_trajectory = None
        best_evaluation = None
        logger.warning("No trajectories to evaluate")

    return agent_weights, {
        'scores': trajectory_scores,
        'best_idx': best_idx,
        'best_trajectory': best_trajectory,
        'best_evaluation': best_evaluation,
        'all_evaluations': detailed_evaluations
    }


def generate_stakeholder_weight_table(trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions,
                                      reasons_cyclist_comfort, reasons_driver_time_eff,
                                      reasons_policymaker_reg_compliance, time_elapsed_driver, time_passed_cyclist,
                                      weight_step=0.1, save_path=None):
    """
    Generate a comprehensive table of trajectory scores for different stakeholder weight combinations.

    Args:
        trajectories_full: List of candidate trajectories
        moving_obstacles: List of moving obstacles (bicycle)
        state: Current state of the vehicle
        car_dimensions: Dimensions of the car
        bicycle_dimensions: Dimensions of the bicycle
        reasons_cyclist_comfort: Initial cyclist comfort score
        reasons_driver_time_eff: Initial driver time efficiency score
        reasons_policymaker_reg_compliance: Initial policymaker regulatory compliance score
        weight_step: Step size for weight variations (default: 0.1)
        save_path: Optional path to save the results to a CSV file

    Returns:
        tuple: Lists of policy_data, driver_data, and cyclist_data for ternary plotting
    """
    import numpy as np
    import pandas as pd
    import time

    # Create empty lists to store results for each category
    policy_data = []
    driver_data = []
    cyclist_data = []

    # Create a fine-grained grid across the entire ternary space
    # Precision for comparison and rounding (to avoid floating point issues)
    precision = max(int(-np.log10(weight_step)) + 2, 6)

    # Generate all possible combinations across the entire ternary space
    combinations = []
    weight_values = np.arange(0, 1.0 + weight_step / 2, weight_step)

    # First generate all possible pairs of (policy, driver) weights
    for policy_w in weight_values:
        for driver_w in weight_values:
            # Calculate the cyclist weight
            cyclist_w = round(1.0 - policy_w - driver_w, precision)

            # Check if this is a valid combination (cyclist weight between 0 and 1)
            if 0 <= cyclist_w <= 1.0 + 1e-9:
                # Store the combination with full precision
                combinations.append((round(policy_w, precision),
                                     round(driver_w, precision),
                                     cyclist_w))

    # Count unique valid combinations
    unique_combinations = set()
    for p, d, c in combinations:
        # Round to ensure consistent handling of floating point values
        unique_combinations.add((round(p, precision), round(d, precision), round(c, precision)))

    print(f"Generating scores for {len(unique_combinations)} weight combinations...")
    print(f"Using weight step: {weight_step}")

    # Progress tracking
    progress_count = 0
    start_time = time.time()
    total_combinations = len(unique_combinations)

    # Process all weight combinations
    for policy_w, driver_w, cyclist_w in unique_combinations:
        # Skip invalid combinations (those that don't sum to 1)
        if abs(policy_w + driver_w + cyclist_w - 1.0) > 1e-6:
            continue

        # Evaluate trajectories with current weights
        results = evaluate_trajectories_with_weights(
            trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions,
            reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
            policy_w, driver_w, cyclist_w,
            time_elapsed_driver, time_passed_cyclist
        )

        # Extract scores
        traj_scores = results['scores']

        # Ensure scores are between 0 and 1
        # First, find and fix negative scores
        for i in range(len(traj_scores)):
            if traj_scores[i] < 0:
                traj_scores[i] = 0.0

        # Then normalize if any score exceeds 1.0
        max_score = max(traj_scores) if traj_scores else 0.0
        if max_score > 1.0:
            traj_scores = [score / max_score for score in traj_scores]

        # Find best trajectory
        if traj_scores:
            max_score = max(traj_scores)
            best_indices = [i for i, score in enumerate(traj_scores) if abs(score - max_score) < 0.000001]

            # Create best trajectory label
            if len(best_indices) == 0:
                best_traj_label = "No valid trajectories    "
            elif len(best_indices) == 1:
                best_traj_label = f"Traj {best_indices[0]}"
            elif len(best_indices) == 2:
                best_traj_label = f"Traj {best_indices[0]} and Traj {best_indices[1]}"
            elif len(best_indices) == 3:
                best_traj_label = f"Traj {best_indices[0]}, Traj {best_indices[1]}, and Traj {best_indices[2]}"
            else:
                best_traj_label = f"Traj {best_indices[0]}, Traj {best_indices[1]}, Traj {best_indices[2]}, and Traj {best_indices[3]}"
        else:
            best_traj_label = "No valid trajectories"

        # Create data row with full precision for internal use
        data_row = [
            policy_w, driver_w, cyclist_w,
            traj_scores[0] if len(traj_scores) > 0 else 0.0,
            traj_scores[1] if len(traj_scores) > 1 else 0.0,
            traj_scores[2] if len(traj_scores) > 2 else 0.0,
            traj_scores[3] if len(traj_scores) > 3 else 0.0,
            best_traj_label
        ]

        # Add to the appropriate dataset based on dominant weight
        if max(policy_w, driver_w, cyclist_w) == policy_w or (policy_w == driver_w and policy_w == cyclist_w):
            policy_data.append(data_row)
        elif max(policy_w, driver_w, cyclist_w) == driver_w:
            driver_data.append(data_row)
        else:  # cyclist is dominant
            cyclist_data.append(data_row)

        # Update progress
        progress_count += 1
        if progress_count % max(1, total_combinations // 20) == 0 or progress_count == total_combinations:
            elapsed_time = time.time() - start_time
            est_total_time = elapsed_time / progress_count * total_combinations
            remaining_time = est_total_time - elapsed_time
            print(f"Progress: {progress_count}/{total_combinations} combinations "
                  f"({progress_count / total_combinations * 100:.1f}%) - "
                  f"ETA: {remaining_time / 60:.1f} minutes")

    # Sort each dataset for a clean presentation
    # Policy data - sort by policy weight first
    policy_data = sorted(policy_data, key=lambda x: (round(x[0], precision), round(x[1], precision)))

    # Driver data - sort by driver weight first
    driver_data = sorted(driver_data, key=lambda x: (round(x[1], precision), round(x[0], precision)))

    # Cyclist data - sort by cyclist weight first
    cyclist_data = sorted(cyclist_data, key=lambda x: (round(x[2], precision), round(x[0], precision)))

    # Save to CSV if path is provided
    if save_path:
        import pandas as pd
        # Combine all data for saving
        all_data = policy_data + driver_data + cyclist_data

        df = pd.DataFrame(all_data, columns=[
            'policy_w', 'driver_w', 'cyclist_w',
            'Traj 0', 'Traj 1', 'Traj 2', 'Traj 3',
            'best_traj_label'
        ])
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    # Format and save the data in code format
    formatted_output = format_weight_data_for_code(policy_data, driver_data, cyclist_data, precision)
    if save_path:
        txt_path = save_path.replace('.csv', '_formatted.txt')
        with open(txt_path, 'w') as f:
            f.write(formatted_output)
        print(f"Formatted code saved to {txt_path}")

    return policy_data, driver_data, cyclist_data


def format_weight_data_for_code(policy_data, driver_data, cyclist_data, precision=3):
    """Format weight data as Python code for direct use"""

    def format_data_rows(data_list):
        formatted_rows = []
        for row in data_list:
            # Format with proper precision
            format_str = f"{{:.{precision}f}}"
            row_str = f"    [{format_str.format(row[0])}, {format_str.format(row[1])}, {format_str.format(row[2])}, " + \
                      f"{format_str.format(row[3])}, {format_str.format(row[4])}, {format_str.format(row[5])}, {format_str.format(row[6])}, " + \
                      f"\"{row[7]}\"]"
            formatted_rows.append(row_str)
        return ",\n".join(formatted_rows)

    policy_formatted = format_data_rows(policy_data)
    driver_formatted = format_data_rows(driver_data)
    cyclist_formatted = format_data_rows(cyclist_data)

    return f"""# Policymaker sensitivity analysis
    policy_data = [
    {policy_formatted}
    ]

    # Driver sensitivity analysis
    driver_data = [
    {driver_formatted}
    ]

    # Cyclist sensitivity analysis
    cyclist_data = [
    {cyclist_formatted}
    ]"""


def evaluate_trajectories_with_weights(trajectories_full, moving_obstacles, state, car_dimensions, bicycle_dimensions,
                                       reasons_cyclist_comfort, reasons_driver_time_eff,
                                       reasons_policymaker_reg_compliance,
                                       policymaker_weight, driver_weight, cyclist_weight,
                                       time_elapsed_driver=0.0, time_passed_cyclist=0.0):
    """
    Evaluate multiple trajectories based on human-centered reasons with custom weights.

    Args:
        trajectories_full: List of candidate trajectories
        moving_obstacles: List of moving obstacles (bicycle)
        state: Current state of the vehicle
        car_dimensions: Dimensions of the car
        bicycle_dimensions: Dimensions of the bicycle
        reasons_cyclist_comfort: Initial cyclist comfort score
        reasons_driver_time_eff: Initial driver time efficiency score
        reasons_policymaker_reg_compliance: Initial policymaker regulatory compliance score
        policymaker_weight: Weight for policymaker score (0.0-1.0)
        driver_weight: Weight for driver score (0.0-1.0)
        cyclist_weight: Weight for cyclist score (0.0-1.0)
        time_elapsed_driver: Initial time elapsed for driver
        time_passed_cyclist: Initial time passed for cyclist

    Returns:
        dict: Evaluation results with scores and best trajectory
    """
    import numpy as np

    # Check for special case of all weights being 0 (edge case)
    if policymaker_weight == 0.0 and driver_weight == 0.0 and cyclist_weight == 0.0:
        # Return zero scores for all trajectories
        num_trajectories = len(trajectories_full)
        return {
            'scores': [0.0] * num_trajectories,
            'best_idx': 0,  # Default to first trajectory
            'best_trajectory': trajectories_full[0] if num_trajectories > 0 else None,
            'best_evaluation': None,
            'all_evaluations': []
        }

    trajectory_scores = []
    detailed_evaluations = []

    # For each trajectory option
    for i, trajectory in enumerate(trajectories_full):
        # 1. Calculate time needed to complete this trajectory
        # For the last trajectory, we need to calculate resampled_trajectory differently
        if i == len(trajectories_full) - 1:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0], last_index=True)
            # set completion time as the real completion time from other trajectories
            # completion_time = calculate_trajectory_completion_time(resampled_trajectory, state, last_index=True)
        else:
            resampled_trajectory = compute_predicted_trajectory(state, trajectory[0])
            completion_time = calculate_trajectory_completion_time(resampled_trajectory, state)

        # 2. Predict bicycle movement for this duration
        bicycle_future_trajectory = np.vstack(MovingObstaclesPrediction(
            *moving_obstacles[0].get(),
            sample_time=ScenarioParameters.DT,
            car_dimensions=bicycle_dimensions
        ).state_prediction(completion_time)).T

        # 3. Sample points along both trajectories
        num_sample_points = len(resampled_trajectory)

        # Create evenly spaced indices
        ego_indices = np.linspace(0, len(resampled_trajectory) - 1, num_sample_points, dtype=int)
        # Delete the last point of bicycle_future_trajectory
        bicycle_indices = np.linspace(0, len(bicycle_future_trajectory[:-1]) - 1, num_sample_points, dtype=int)

        # 4. Evaluate reasons at each sample point
        current_time_elapsed_driver = time_elapsed_driver
        current_time_passed_cyclist = time_passed_cyclist

        policymaker_scores = []
        driver_scores = []
        cyclist_comfort_scores = []
        cyclist_time_scores = []
        cyclist_combined_scores = []

        # Width of the car for centerline evaluation
        car_width = car_dimensions.bounding_box_size[0]

        for j in range(num_sample_points):
            # Get ego vehicle state at this point
            ego_x = resampled_trajectory[ego_indices[j], 0]
            ego_y = resampled_trajectory[ego_indices[j], 1]
            ego_theta = resampled_trajectory[ego_indices[j], 2]

            # Get bicycle state at this point
            bicycle_x = bicycle_future_trajectory[bicycle_indices[j], 0]
            bicycle_y = bicycle_future_trajectory[bicycle_indices[j], 1]

            # Create a simulated state object for evaluation
            simulated_state = State(x=ego_x, y=ego_y, yaw=ego_theta, v=state.v)

            # Create a simulated obstacle object for evaluation
            simulated_obstacle = type('obj', (object,), {
                'get': lambda self=None: (bicycle_x, bicycle_y, 0, 0, CyclistParameters.SPEED, 0)
            })
            simulated_obstacles = [simulated_obstacle]

            # Evaluate policymaker (regulatory compliance)
            policymaker_score = evaluate_distance_to_centerline(
                ego_x, car_width, ScenarioParameters.CENTERLINE_LOCATION)
            policymaker_scores.append(policymaker_score)

            # Evaluate driver time efficiency
            driver_score, current_time_elapsed_driver = evaluate_time_following(
                'driver_reasons', ScenarioParameters.DT,
                DriverParameters.DISTANCE_BUFFER, DriverParameters.DISTANCE_REF,
                DriverParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_elapsed_driver)
            driver_scores.append(driver_score)

            # Evaluate cyclist comfort (distance)
            cyclist_comfort_score = evaluate_distance_to_obstacle(
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                simulated_obstacles, simulated_state)
            cyclist_comfort_scores.append(cyclist_comfort_score)

            # Evaluate cyclist comfort (time)
            cyclist_time_score, current_time_passed_cyclist = evaluate_time_following(
                'cyclist_reasons', ScenarioParameters.DT,
                CyclistParameters.DISTANCE_BUFFER, CyclistParameters.DISTANCE_REF,
                CyclistParameters.TIME_THRESHOLD, simulated_obstacles,
                simulated_state, current_time_passed_cyclist)
            cyclist_time_scores.append(cyclist_time_score)

            # Combined cyclist score
            cyclist_combined_score = cyclist_comfort_score * cyclist_time_score
            cyclist_combined_scores.append(cyclist_combined_score)

        # delete last value of scores because the ego vehicle is not moving but the bicycle is
        policymaker_scores = policymaker_scores[:-1]
        driver_scores = driver_scores[:-1]
        cyclist_combined_scores = cyclist_combined_scores[:-1]

        # replace first value of reasons scores with the first value of reasons evaluation
        policymaker_scores[0] = reasons_policymaker_reg_compliance
        driver_scores[0] = reasons_driver_time_eff
        cyclist_combined_scores[0] = reasons_cyclist_comfort

        # 5. Calculate average scores for each reason
        avg_policymaker = np.mean(policymaker_scores)
        avg_driver = np.mean(driver_scores)
        avg_cyclist = np.mean(cyclist_combined_scores)

        # 6. Calculate weighted total score
        # Use custom weights provided as parameters
        agent_weights = {
            'policymaker': policymaker_weight,  # Regulatory compliance
            'driver': driver_weight,  # Driver patience/efficiency
            'cyclist': cyclist_weight  # Cyclist comfort/safety
        }
        # ideal_weights = [0.44, 0.25, 0.31]  # Cyclist, Driver, Policymaker -> more focus on driver, based on interview results
        # ideal_weights = [0.1, 0.1, 0.8]  # Cyclist, Driver, Policymaker -> more focus on policymaker
        ideal_weights = [1 / 3, 1 / 3, 1 / 3]  # Cyclist, Driver, Policymaker -> equal focus

        # Calculate balance function value - ensure it returns 1.0 for special cases
        balance_value = 1.0
        if sum(agent_weights.values()) > 0:
            balance_value = balance_function([agent_weights.get('cyclist', 0),
                                              agent_weights.get('driver', 0),
                                              agent_weights.get('policymaker', 0)],
                                             ideal_weights)

        total_score = balance_value * (
                agent_weights['policymaker'] * avg_policymaker +
                agent_weights['driver'] * avg_driver +
                agent_weights['cyclist'] * avg_cyclist
        )

        # Ensure score is capped at 1.0 and not negative
        total_score = max(0.0, min(total_score, 1.0))

        # Store results
        trajectory_scores.append(total_score)

        detailed_eval = {
            'trajectory_idx': i,
            'total_score': total_score,
            'completion_time': completion_time,
            'avg_scores': {
                'policymaker': avg_policymaker,
                'driver': avg_driver,
                'cyclist': avg_cyclist
            },
            'detailed_scores': {
                'policymaker': policymaker_scores,
                'driver': driver_scores,
                'cyclist_comfort': cyclist_comfort_scores,
                'cyclist_time': cyclist_time_scores,
                'cyclist_combined': cyclist_combined_scores
            }
        }

        detailed_evaluations.append(detailed_eval)

        logger.info(f"Trajectory {i}: Score {total_score:.3f} (P: {avg_policymaker:.2f}, "
                    f"D: {avg_driver:.3f}, C: {avg_cyclist:.3f}) "
                    f"Time: {completion_time:.3f}s")

    # 7. Find the best trajectory
    if trajectory_scores:
        best_idx = np.argmax(trajectory_scores)
        best_score = trajectory_scores[best_idx]
        best_trajectory = trajectories_full[best_idx]
        best_evaluation = detailed_evaluations[best_idx]

        logger.info(f"Selected trajectory {best_idx} with score {best_score:.3f}")
    else:
        best_idx = None
        best_trajectory = None
        best_evaluation = None
        logger.warning("No trajectories to evaluate")

    return {
        'scores': trajectory_scores,
        'best_idx': best_idx,
        'best_trajectory': best_trajectory,
        'best_evaluation': best_evaluation,
        'all_evaluations': detailed_evaluations
    }


def calculate_trajectory_completion_time(trajectory_res, state, last_index=None):
    """
    Calculate the estimated time to complete a trajectory based on vehicle dynamics.

    Args:
        trajectory_res: The trajectory to analyze
        state: The current state of the vehicle

    Returns:
        float: Estimated time to complete the trajectory in seconds
    """
    # Start with current velocity
    current_velocity = state.v

    # Initialize time counter
    total_time = 0.0

    # Calculate distance between consecutive points
    if len(trajectory_res) <= 1:
        return 0.0

    distances = []
    for i in range(1, len(trajectory_res)):
        # Calculate Euclidean distance between consecutive points
        distance = np.linalg.norm(trajectory_res[i, :2] - trajectory_res[i - 1, :2])
        distances.append(distance)

    # Simulate movement along the trajectory
    for distance in distances:
        # Update velocity based on acceleration
        if last_index == None:
            current_velocity = min(current_velocity + MAX_ACCEL, Simulation.MAX_SPEED)
        else:
            current_velocity = current_velocity
        # Calculate time needed for this segment
        segment_time = distance / current_velocity
        total_time += segment_time

    return total_time


def reasons_evaluation(reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance,
                       replan_needed, replan_tracker):
    """
    Check if any reason is below the threshold and trigger replanning if needed.

    Args:
        reasons_policymaker_reg_compliance: Policymaker compliance reason.
        reasons_driver_time_eff: Driver time efficiency reason.
        reasons_cyclist_comfort: Cyclist comfort reason.
        replan_tracker: Whether a replan has already been triggered.
        replan_needed: Whether a replan is needed.

    Returns:
        tuple: A tuple containing whether replan is needed and the updated replan tracker.
    """
    reasons = {
        "Policymaker Compliance": reasons_policymaker_reg_compliance,
        "Driver Time Efficiency": reasons_driver_time_eff,
        "Cyclist Comfort": reasons_cyclist_comfort,
    }

    # Check if any reason is below the threshold
    if any(value < ReasonParameters.REASONS_THRESHOLD for value in reasons.values()):
        if not replan_tracker:
            for reason_name, value in reasons.items():
                if value < ReasonParameters.REASONS_THRESHOLD:
                    logger.info(f'Reasons below {ReasonParameters.REASONS_THRESHOLD * 100}%, {reason_name}, Replan')
            replan_tracker = True
            replan_needed = True
    else:
        replan_tracker = False
        replan_needed = False

    return replan_needed, replan_tracker


def run_motion_primitive_search(scenario_no_obstacles, car_dimensions, mps, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2,
                                yyy3, zzz0, zzz1, zzz2, zzz3) -> tuple:
    """
    Run the motion primitive search algorithm.

    Args:
        scenario_no_obstacles: The scenario without obstacles.
        car_dimensions: Dimensions of the car.
        mps: Motion primitives.

    Returns:
        tuple: A tuple containing the cost, path, and trajectory.
    """
    start_time = time.time()
    search = MotionPrimitiveSearch(scenario_no_obstacles, car_dimensions, mps, xxx0=xxx0, xxx1=xxx1, xxx2=xxx2,
                                   xxx3=xxx3, yyy0=yyy0, yyy1=yyy1, yyy2=yyy2, yyy3=yyy3, zzz0=zzz0, zzz1=zzz1,
                                   zzz2=zzz2, zzz3=zzz3, margin=car_dimensions.radius)
    cost, path, trajectory_full = search.run(debug=True)
    logger.info("Search finished")
    plot_motion_primitives(search, scenario_no_obstacles, path, car_dimensions)
    end_time = time.time()
    search_runtime = end_time - start_time
    logger.info(f"Search runtime: {search_runtime}")
    return cost, path, trajectory_full, search_runtime


def initialize_simulation() -> tuple:
    """
    Initialize simulation parameters, motion primitives, and the scenario.

    Returns:
        tuple: A tuple containing initialized objects and parameters.
    """
    # Load motion primitives
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)
    bicycle_dimensions = BicycleRealDimensions(skip_back_circle_collision_checking=False)

    # Define the scenario
    arterial = ArterialMultiLanes(num_lanes=2, goal_lane=1)
    scenario_no_obstacles = arterial.create_scenario()
    scenario_visualization = arterial.create_scenario(frame_visualization=True)

    # Define moving obstacles
    spawn_location_x = scenario_no_obstacles.start[0] + ScenarioParameters.X_LOC_CYCLIST_BUFFER
    spawn_location_y = scenario_no_obstacles.start[1] + ScenarioParameters.Y_LOC_CYCLIST_BUFFER
    moving_obstacles = [
        MovingObstacleArterial(bicycle_dimensions, spawn_location_x, spawn_location_y, speed=CyclistParameters.SPEED,
                               initial_speed=CyclistParameters.SPEED, offset=True, dt=ScenarioParameters.DT)
    ]

    return mps, car_dimensions, bicycle_dimensions, arterial, scenario_no_obstacles, scenario_visualization, moving_obstacles


def initialize_mpc(trajectory_full, car_dimensions, max_speed) -> tuple:
    """
    Initialize the MPC controller.

    Args:
        trajectory_full: The full trajectory from the motion primitive search.
        car_dimensions: Dimensions of the car.

    Returns:
        tuple: A tuple containing the MPC object, the initial state and the distance between the points in the trajectory.
    """
    dl = np.linalg.norm(trajectory_full[0, :2] - trajectory_full[1, :2])
    mpc = MPC(cx=trajectory_full[:, 0], cy=trajectory_full[:, 1], cyaw=trajectory_full[:, 2], dl=dl,
              dt=ScenarioParameters.DT, car_dimensions=car_dimensions, speed=max_speed)
    state = State(x=trajectory_full[0, 0], y=trajectory_full[0, 1], yaw=trajectory_full[0, 2],
                  v=CyclistParameters.SPEED)
    return mpc, state, dl


def evaluate_reasons(state, moving_obstacles, car_dimensions, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST) -> tuple:
    """
    Evaluate the reasons for policymaker, driver, and cyclist.

    Args:
        state: Current state of the vehicle.
        moving_obstacles: List of moving obstacles.
        car_dimensions: Dimensions of the car.
        TIME_ELAPSED_DRIVER: Time elapsed for the driver.
        TIME_PASSED_CYCLIST: Time passed for the cyclist.

    Returns:
        tuple: A tuple containing the evaluated reasons and updated times.
    """
    car_width = car_dimensions.bounding_box_size[0]
    reasons_policymaker_reg_compliance = evaluate_distance_to_centerline(state.x, car_width,
                                                                         ScenarioParameters.CENTERLINE_LOCATION)
    reasons_driver_time_eff, TIME_ELAPSED_DRIVER = evaluate_time_following('driver_reasons', ScenarioParameters.DT,
                                                                           DriverParameters.DISTANCE_BUFFER,
                                                                           DriverParameters.DISTANCE_REF,
                                                                           DriverParameters.TIME_THRESHOLD,
                                                                           moving_obstacles, state, TIME_ELAPSED_DRIVER)
    reasons_cyclist_time_eff, TIME_PASSED_CYCLIST = evaluate_time_following('cyclist_reasons', ScenarioParameters.DT,
                                                                            CyclistParameters.DISTANCE_BUFFER,
                                                                            CyclistParameters.DISTANCE_REF,
                                                                            CyclistParameters.TIME_THRESHOLD,
                                                                            moving_obstacles, state,
                                                                            TIME_PASSED_CYCLIST)
    reasons_cyclist_distance = evaluate_distance_to_obstacle(CyclistParameters.DISTANCE_BUFFER,
                                                             CyclistParameters.DISTANCE_REF, moving_obstacles, state)
    reasons_cyclist_comfort = reasons_cyclist_time_eff * reasons_cyclist_distance
    return reasons_policymaker_reg_compliance, reasons_driver_time_eff, reasons_cyclist_comfort, TIME_ELAPSED_DRIVER, TIME_PASSED_CYCLIST


def plot_trajectories(obstacles_positions, ego_positions: History):
    # Create a new figure and get the current axes
    fig = plt.figure()
    ax = plt.gca()

    # Get colormap
    cmap = plt.cm.get_cmap('viridis')

    # Time step duration
    dt = 0.2  # seconds

    # For the ego vehicle
    times, ego_x, ego_y = ego_positions.t, ego_positions.x, ego_positions.y
    ego_positions = np.column_stack((ego_x, ego_y))
    # Convert time to seconds and normalize for color mapping
    times = np.array(times) * dt  # convert times to a numpy array and to seconds
    times_norm = times / max(times)
    # Plot each segment of the trajectory with a color corresponding to its time
    for i_time in range(1, len(times)):
        color = cmap(times_norm[i_time])
        plt.plot(ego_positions[(i_time - 1):(i_time + 1), 0], ego_positions[(i_time - 1):(i_time + 1), 1], color=color,
                 linewidth=8)

    # For each obstacle
    for i_obstacle, obstacle_positions in enumerate(obstacles_positions):
        # Unpack the positions and times
        times, positions = zip(*obstacle_positions)
        positions = np.array(positions)
        # Convert time to seconds and normalize for color mapping
        times = np.array(times) * dt  # convert times to a numpy array and to seconds
        times_norm = times / max(times)
        # Plot each segment of the trajectory with a color corresponding to its time
        for i_time in range(1, len(times)):
            color = cmap(times_norm[i_time])
            plt.plot(positions[(i_time - 1):(i_time + 1), 0], positions[(i_time - 1):(i_time + 1), 1], color=color,
                     linewidth=4)

    # Add a colorbar indicating the time progression in seconds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(times)))
    cb = fig.colorbar(sm, ax=ax)  # add the colorbar to the current axes

    # Set colorbar ticks to every 2 seconds
    tick_locator = ticker.MultipleLocator(base=2)
    cb.locator = tick_locator
    cb.update_ticks()

    # Set colorbar label font size
    cb.set_label('Time (seconds)', size=12)

    # Set colorbar tick label font size
    cb.ax.tick_params(labelsize=10)

    # Set labels and title with smaller font size
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.title('Trajectories of Moving Obstacles', fontsize=12)

    # Set axis limits
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    # Reduce tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Show the plot
    plt.show()


def plot_motion_primitives(search, scenario, path, car_dimensions):
    fig, ax = plt.subplots()
    draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)

    # Plot scenario with car
    car_state = State(x=scenario.start[0], y=scenario.start[1], yaw=scenario.start[2], v=0.0)
    plot_scenario_with_car(scenario, car_state, car_dimensions, ax=ax)
    plot_path(path, ax)
    plt.show()


def plot_path(path, ax):
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], label='Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Path')
    # ax.legend()
    ax.grid(True)
    ax.axis('equal')


def plot_scenario_with_car(scenario, car_state, car_dimensions, ax):
    # Plot obstacles
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='b')

    # Plot car
    draw_car((car_state.x, car_state.y, car_state.yaw), car_dimensions, ax=ax, color='r')

    # Plot goal area
    goal_area = scenario.goal_area
    goal_area.draw(ax, color='g')

    plt.title('Scenario with Car Location')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_deviation(ax, time_values, xref_deviation_value):
    fontsize = 25
    ax.clear()
    ax.plot(time_values, xref_deviation_value, "-r", label="Deviation from reference trajectory")
    ax.grid(False)
    ax.set_xlabel("Time [s]", fontsize=fontsize)
    ax.set_ylabel("Deviation [m]", fontsize=fontsize)
    ax.set_ylim(0, 0.035)  # Set the y-axis limit


def visualize_final(history: History):
    fontsize = 25

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, np.array(history.v) * 3.6, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Speed [km/h]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, history.a, "-r", label="acceleration")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Acceleration [$m/s^2$]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.rcParams['font.size'] = fontsize
    plt.plot(history.t, history.xref_deviation, "-r", label="Deviation from reference trajectory")
    plt.grid(True)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Deviation [m]", fontsize=fontsize)
    plt.tight_layout()
    plt.show()


# Dictionary to store moving obstacle history
obstacle_history = {}  # Key: obstacle, Value: list of (x, y, theta) over time


def setup_plot(ax, tmp_trajectory, collision_xy, state, trajectory_res):
    """Set up the basic plot elements."""
    ax.set_facecolor('#AFABAB')
    ax.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='b')

    if collision_xy is not None:
        ax.scatter([collision_xy[0]], [collision_xy[1]], color='r')

    ax.scatter([state.x], [state.y], color='r')
    ax.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')


def draw_static_elements(ax, scenario):
    """Draw static obstacles and vertical lines."""
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='#9ED386')

    # Vertical lines
    ax.axvline(x=0 + 0.3, color='#FFBD00')
    ax.axvline(x=0 - 0.3, color='#FFBD00')
    ax.axvline(x=ScenarioParameters.WIDTH_ROAD - 0.2, color='#FFFFFF')
    ax.axvline(x=ScenarioParameters.WIDTH_ROAD + 0.2, color='#FFFFFF')


def update_obstacle_history(moving_obstacles, obstacle_history):
    """Update the history of moving obstacles."""
    for mo in moving_obstacles:
        x, y, _, theta, _, _ = mo.get()  # Get current state

        if mo not in obstacle_history:  # First time seeing this obstacle
            obstacle_history[mo] = []

        obstacle_history[mo].append((x, y, theta))  # Store position


def plot_historical_data(ax, simulation, mpc, car_dimensions, bicycle_dimensions, obstacle_history, plot_times, dt):
    """Plot historical data for the car and moving obstacles."""
    for plot_time in plot_times:
        time_index = int(plot_time / dt)  # Convert time to index in history
        if time_index < len(simulation.history.x):  # Ensure index is within bounds
            x, y, yaw = (simulation.history.x[time_index],
                         simulation.history.y[time_index],
                         simulation.history.yaw[time_index])

            draw_car((x, y, yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=ax,
                     color='black', draw_collision_circles=False)

            # Label the time step **to the left** of the car
            ax.text(x - 2.0, y + 0.5, f"{plot_time}s", fontsize=12, color='black', ha='center')

        # Plot historical bicycles (moving obstacles)
        for mo in obstacle_history:
            if time_index < len(obstacle_history[mo]):
                x, y, theta = obstacle_history[mo][time_index]  # Retrieve past position

                draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                             draw_collision_circles=False, color='blue')

                # Label the time step **to the right** of the bicycle
                ax.text(x + 1.5, y + 0.5, f"{plot_time}s", fontsize=12, color='blue', ha='center')


def plot_current_state(ax, state, mpc, car_dimensions, moving_obstacles, bicycle_dimensions):
    """Plot the current state of the car and moving obstacles."""
    draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=ax,
             color='black', draw_collision_circles=False)

    for mo in moving_obstacles:
        x, y, _, theta, _, _ = mo.get()  # Get current position
        draw_bicycle((x, y, theta), bicycle_dimensions, ax=ax,
                     draw_collision_circles=False, color='blue')


def finalize_plot(ax, simulation, mpc, i, dt):
    """Add final touches to the plot (trajectory, labels, legend, etc.)."""
    ax.plot(simulation.history.x, simulation.history.y, '--k', alpha=0.5)

    if mpc.ox is not None:
        ax.plot(mpc.ox, mpc.oy, "+r")

    ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k")

    # Legend for clarity
    # ax.legend(fontsize=12, loc="upper right")

    # Axis labels and formatting
    ax.set_title(f"Time: {i * dt:.2f} [s]", fontsize=20)
    # ax.set_xlabel('X', fontsize=20)
    # ax.set_ylabel('Y', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.axis("equal")
    ax.grid(True)
    # ax.set_xlim((-10, 10))
    ax.set_ylim((-30, 30))

    # remove xticks and yticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_car_and_obstacles(ax, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                           simulation, mpc, car_dimensions, bicycle_dimensions, i, dt, historical_plot=True):
    """Main function to plot the car and obstacles."""
    global obstacle_history  # Use global variable to track history across function calls

    # Set up the plot
    setup_plot(ax, tmp_trajectory, collision_xy, state, trajectory_res)

    # Draw static elements
    draw_static_elements(ax, scenario)

    # Update obstacle history
    update_obstacle_history(moving_obstacles, obstacle_history)

    # Plot simulation history
    ax.plot(simulation.history.x, simulation.history.y, '-r')

    if mpc.ox is not None:
        ax.plot(mpc.ox, mpc.oy, "+r")

    ax.plot(mpc.xref[0, :], mpc.xref[1, :], "+k")

    # Determine time steps for historical plotting
    time_elapsed = i * dt  # Current simulation time
    final_time = time_elapsed  # End of simulation time

    # Generate time points: 0, 5, 10, ..., final_time (if not already included)
    plot_times = list(range(0, int(final_time) + 1, 5))  # Every 5 seconds
    if final_time not in plot_times:  # Ensure last time step is included
        plot_times.append(int(final_time))

    if historical_plot:
        plot_historical_data(ax, simulation, mpc, car_dimensions, bicycle_dimensions, obstacle_history, plot_times, dt)
    else:
        plot_current_state(ax, state, mpc, car_dimensions, moving_obstacles, bicycle_dimensions)

    # Finalize the plot
    finalize_plot(ax, simulation, mpc, i, dt)


def plot_reasons(ax, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values):
    # Plot reasons values over time
    # Define colors for better balance
    colors = ['dodgerblue', 'darkorange', 'mediumseagreen']

    ax.clear()  # Clear the axis for each new update

    ignorethis = '''
    ax.plot(time_values, reasons_policymaker_values, label=r'$R_{policymaker}$', color=colors[0],
             linestyle='--', linewidth=2);
    ax.plot(time_values, reasons_driver_values, label=r'$R_{driver}$', color=colors[1], linestyle='--',
             linewidth=2);
    ax.plot(time_values, reasons_cyclist_values, label=r'$R_{cyclist}$', color=colors[2], linestyle='--',
             linewidth=2);
    '''
    ax.plot(time_values, reasons_policymaker_values, label=r'Regulatory compliance', color=colors[0],
            linestyle='--', linewidth=2)
    ax.plot(time_values, reasons_driver_values, label=r"Driver's patience", color=colors[1], linestyle='--',
            linewidth=2)
    ax.plot(time_values, reasons_cyclist_values, label=r"Cyclist's safety and comfort", color=colors[2], linestyle='--',
            linewidth=2)

    ax.axhline(y=ReasonParameters.REASONS_THRESHOLD, color='red', linestyle=':', linewidth=2, label='Replan Threshold')
    # Annotate the threshold line
    # Annotate the threshold line with multiple lines of text
    ax.text(time_values[0] + 0.2, ReasonParameters.REASONS_THRESHOLD + 0.05,
            "If reasons below \nthreshold, replan",
            color='red', fontsize=11, verticalalignment='bottom')

    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Reasons [0-1]', fontsize=25)
    # ax.set_title('Reasons Values Over Time', fontsize=20)
    ax.set_ylim([0, 1.1])
    # ax.legend(fontsize=10, loc='lower left')
    ax.grid(False)


def plot_velocity(ax, time_values, speed_values):
    # Plot reasons values over time
    ax.clear()  # Clear the axis for each new update

    # Plot the distance values
    ax.plot(time_values, speed_values, label='Speed of the Vehicle')

    # Set axis labels and title
    ax.set_xlabel('Time [s]', fontsize=25)
    ax.set_ylabel('Speed [km/h]', fontsize=25)
    ax.set_ylim(0, 35)  # Set the y-axis limit
    # ax.set_title('Speed of the Vehicle', fontsize=20)

    # Add legend
    # ax.legend(fontsize=16)

    # Enable grid
    ax.grid(False)


def plot_distance(ax, time_values, distance_values, DISTANCE_THRESHOLD_CAR, DISTANCE_THRESHOLD_BICYCLE):
    # Clear the axis for each new update
    ax.clear()

    # Plot the distance values
    ax.plot(time_values, distance_values, label='Distance between Car and Bicycle')

    # Define thresholds dynamically based on input values
    thresholds = [
        {'value': DISTANCE_THRESHOLD_CAR, 'color': 'r', 'label': f'{DISTANCE_THRESHOLD_CAR}m Threshold'},
        {'value': DISTANCE_THRESHOLD_BICYCLE, 'color': 'g', 'label': f'{DISTANCE_THRESHOLD_BICYCLE}m Threshold'}
    ]

    # Add vertical lines at the specified thresholds
    for threshold in thresholds:
        for i, dist in enumerate(distance_values):
            if i == 0:  # Skip the first iteration to avoid index issues
                continue
            if dist <= threshold['value'] and distance_values[i - 1] > threshold['value']:
                ax.axvline(
                    time_values[i],
                    color=threshold['color'],
                    linestyle='--',
                    label=threshold['label'] if threshold['label'] not in [line.get_label() for line in
                                                                           ax.get_lines()] else ''
                )

    # Set axis labels and title
    ax.set_xlabel('Time Frame', fontsize=15)
    ax.set_ylabel('Distance', fontsize=15)
    ax.set_title('Distance Between Car and Bicycle', fontsize=18)

    # Add legend
    # ax.legend(fontsize=12)

    # Enable grid
    ax.grid(True)


def visualize_frame(dt, car_dimensions, bicycle_dimensions, collision_xy, i, moving_obstacles, mpc,
                    scenario, simulation, state, tmp_trajectory, trajectory_res,
                    reasons_cyclist_values, reasons_driver_values, reasons_policymaker_values, distance_values,
                    reasons_cyclist_comfort, reasons_driver_time_eff, reasons_policymaker_reg_compliance, speed_values,
                    time_values, xref_deviation_values, xref_deviation_value,
                    static_x_axis=True, max_time=20, save_path=None,
                    historical_plot=False):  # ADDED: save_path parameter
    """
    Visualize the simulation frame with an option for static or dynamic x-axis.

    Parameters:
        static_x_axis (bool): If True, the x-axis is fixed to `max_time`. If False, the x-axis dynamically adjusts.
        max_time (float): Maximum time for the x-axis when `static_x_axis` is True.
        save_path (Path): The Path object for the directory to save the frame.
    """
    # The directory is now passed in as the 'save_path' argument.
    # The os.path.join and os.makedirs calls here are no longer needed, as they
    # are handled by the run_simulation function.

    if i >= 0:
        # Create figure and grid layout (2 rows, 2 columns)
        fig = plt.figure(figsize=(10, 10))  # Adjust size for better visualization
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # First row taller

        # Create subplots
        ax1 = plt.subplot(gs[:, 0])  # Merge (1,1) and (2,1) into a single tall plot
        ax2 = plt.subplot(gs[0, 1])  # Top-right plot
        ax3 = plt.subplot(gs[1, 1])  # Bottom-right plot
        ax4 = plt.subplot(gs[2, 1])

        # Update ax1 with the car, obstacles, and other related information
        plot_car_and_obstacles(ax1, tmp_trajectory, collision_xy, state, trajectory_res, scenario, moving_obstacles,
                               simulation, mpc, car_dimensions, bicycle_dimensions, i, dt,
                               historical_plot=historical_plot)

        # Update ax2 with reasons values over time
        time_value = i * dt  # Time value for current simulation step
        time_values.append(time_value)
        reasons_policymaker_values.append(reasons_policymaker_reg_compliance)
        reasons_driver_values.append(reasons_driver_time_eff)
        reasons_cyclist_values.append(reasons_cyclist_comfort)
        speed_values.append(state.v * 3.6)  # times 3.6 to convert from m/s to km/h
        distance_values.append(np.linalg.norm(
            [moving_obstacles[0].get()[0] - state.x,
             moving_obstacles[0].get()[1] - state.y]))  # Placeholder for distance values
        xref_deviation_values.append(xref_deviation_value)

        # Plot the reasons and velocity in separate subplots
        plot_reasons(ax2, time_values, reasons_policymaker_values, reasons_driver_values, reasons_cyclist_values)
        plot_velocity(ax3, time_values, speed_values)
        plot_deviation(ax4, time_values, xref_deviation_values)

        # Set x-axis limits based on static or dynamic option
        if static_x_axis:
            # Use the predefined max_time for static x-axis
            ax2.set_xlim(0, max_time)
            ax3.set_xlim(0, max_time)
            ax4.set_xlim(0, max_time)
        else:
            # Use the current simulation time for dynamic x-axis
            current_max_time = time_value
            ax2.set_xlim(0, current_max_time)
            ax3.set_xlim(0, current_max_time)
            ax4.set_xlim(0, current_max_time)

        # Adjust layout and show the plot
        plt.tight_layout()

        # CRITICAL CHANGE: Use the save_path argument to save the frame
        if save_path:
            full_save_path = save_path / f"frame_{i:04d}.jpg"
            plt.savefig(full_save_path, format='jpg', dpi=300)

        # plt.pause(0.001)
        plt.close()


def save_vehicle_data(simulation, time_values, reasons_policymaker_values, reasons_driver_values,
                      reasons_cyclist_values, supervision):
    """
    Save vehicle data to a CSV file.
    """
    time_simulation = time_values[:-1]
    acceleration_vehicle = simulation.history.a[:-1]
    velocity_vehicle = simulation.history.v[:-1]
    x_position_vehicle = simulation.history.x[:-1]
    y_position_vehicle = simulation.history.y[:-1]
    yaw_vehicle = simulation.history.yaw[:-1]
    x_ref_deviation_vehicle = simulation.history.xref_deviation[:-1]

    save_folder = os.path.join("main", "results", "reasons_evaluation")
    os.makedirs(save_folder, exist_ok=True)

    if supervision:
        save_path = os.path.join(save_folder, "vehicle_data_spv.csv")
    else:
        save_path = os.path.join(save_folder, "vehicle_data_unspv.csv")

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['time', 'acceleration', 'v', 'x', 'y', 'yaw', 'x_ref_dev', 'reasons_policymaker', 'reasons_driver',
             'reasons_cyclist'])
        for row in zip(time_simulation, acceleration_vehicle, velocity_vehicle, x_position_vehicle, y_position_vehicle,
                       yaw_vehicle, x_ref_deviation_vehicle, reasons_policymaker_values, reasons_driver_values,
                       reasons_cyclist_values):
            writer.writerow(row)
    logger.info("Vehicle data saved")


import os
import subprocess
import streamlit as st
from pathlib import Path
from dataclasses import dataclass

# Assuming main() is already defined somewhere above or imported

# Path to the parameters file
PARAMETERS_PATH = Path(__file__).resolve().parent.parent / "lib" / "parameters.py"


def update_parameters_file(params):
    """
    This function generates and writes a new parameters.py file
    based on the user inputs.
    """
    file_content = f"""# parameters.py - Generated by Streamlit App
from dataclasses import dataclass

@dataclass
class ScenarioParameters:
    DT = {params['scenario_dt']}  # Time step
    CENTERLINE_LOCATION = {params['scenario_centerline_location']}  # Centerline location for evaluation
    LENGTH = {params['scenario_length']}  # Length of the scenario
    WIDTH_ROAD = {params['scenario_width_road']} # Width of a single road lane
    X_LOC_GOAL = {params['scenario_x_loc_goal']} # x location of the goal, right side of the middle of the road
    Y_LOC_GOAL = {params['scenario_y_loc_goal']} # y location of the goal
    X_LOC_EGO = {params['scenario_x_loc_ego']} # Initial x location of the ego vehicle (AV), right side of the middle of the road
    X_LOC_CYCLIST_BUFFER = {params['scenario_x_loc_cyclist_buffer']} # Initial x location of the cyclist
    Y_LOC_CYCLIST_BUFFER = {params['scenario_y_loc_cyclist_buffer']} # Initial x location of the cyclist

@dataclass
class ReasonParameters:
    REASONS_THRESHOLD = {params['reasons_threshold']}  # Threshold for reasons to trigger replan

@dataclass
class MPCParameters:
    TIME_HORIZON = {params['mpc_time_horizon']}  # Time horizon for predictions
    FRAME_WINDOW = {params['mpc_frame_window']}  # Frame window for collision checking
    MAX_SPEED_FREEWAY = {params['mpc_max_speed_freeway']}  # Maximum speed for freeway

@dataclass
class DriverParameters:
    DISTANCE_REF = {params['driver_distance_ref']}  # Reference distance for driver patience
    DISTANCE_BUFFER = {params['driver_distance_buffer']}  # Buffer distance for driver patience
    TIME_THRESHOLD = {params['driver_time_threshold']}  # Time threshold for driver patience

@dataclass
class CyclistParameters:
    DISTANCE_REF = {params['cyclist_distance_ref']}  # Reference distance for cyclist patience
    DISTANCE_BUFFER = {params['cyclist_distance_buffer']}  # Buffer distance for cyclist patience
    TIME_THRESHOLD = {params['cyclist_time_threshold']}  # Time threshold for cyclist patience
    SPEED = {params['cyclist_speed']} # Cyclist speed [km/h]
"""
    with open(PARAMETERS_PATH, "w") as f:
        f.write(file_content)


def run_simulation(weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist, ideal_weight_driver,
                   ideal_weight_policymaker, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2, yyy3, zzz0, zzz1, zzz2, zzz3):
    # Use pathlib to get the directory of the current file
    script_dir = Path(__file__).parent

    # Correctly go up TWO levels to the root of the repo
    grandparent_dir = script_dir.parent.parent
    results_folder = grandparent_dir / "results" / "reasons_evaluation"

    # --- THIS LINE IS CRITICAL ---
    # Create the results directory if it does not exist
    results_folder.mkdir(parents=True, exist_ok=True)

    st.info(f"Using results directory: {results_folder}")

    # --- Fix for the main() function's missing path handling ---
    # This is the most likely cause of your problem.
    # The 'main' function must be told where to save the files.
    # If your main() function accepts a path, pass results_folder to it.
    # e.g., main(..., save_path=results_folder)
    # If not, you must fix the code inside main() to use this folder.

    # Run the simulation
    main(weightofpolicy, weightofdriver, weightofcyclist, ideal_weight_cyclist, ideal_weight_driver,
         ideal_weight_policymaker, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2, yyy3, zzz0, zzz1, zzz2, zzz3,
         replanner=True, vis_frame=True, save_weight_table=False, historical_plot=False, save_path=results_folder)

    # --- START OF DIAGNOSTIC BLOCK ---
    st.info("Checking for generated frames...")
    try:
        files_in_dir = os.listdir(results_folder)
        frames_found = [f for f in files_in_dir if f.startswith('frame_')]

        if not frames_found:
            st.error(f"FAILURE: No files starting with 'frame_' were found in: {results_folder}")
            st.info(f"Files found in directory: {files_in_dir}")
        else:
            st.success(f"SUCCESS: Found {len(frames_found)} frames in the directory. Proceeding with video generation.")
            # Only proceed with ffmpeg if files were found
            # Generate output video using ffmpeg
            input_images_path = results_folder / "frame_%04d.jpg"
            output_video_path = results_folder / "output_video.mp4"
            trajectory_evaluation_spatial = results_folder / "trajectory_evaluations_spatial.png"
            trajectory_evaluations_trajectories = results_folder / "trajectory_evaluations_trajectories.png"

            subprocess.run([
                'ffmpeg', '-y', '-framerate', '10', '-i', str(input_images_path),
                '-s', '1280x720',  # Example: downscale to 720p
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '28', str(output_video_path)
            ])
            # Display the video in Streamlit
            st.subheader("⚙ Results ")
            if output_video_path.exists():

                # 1. Create three columns with equal width (1, 1, 1)
                col1, col2, col3 = st.columns(3)

                # --- Column 1: Spatial Trajectory Evaluation ---
                with col1:
                    st.subheader("1. Generated Trajectories from Step 1")
                    # Use st.image for the figure
                    st.image(str(trajectory_evaluation_spatial), caption="Spatial Trajectory Evaluation")

                # --- Column 2: Trajectory Evaluation Results ---
                with col2:
                    st.subheader("2. Evaluation Results of Three Trajectories from Step 2")
                    # Use st.image for the figure
                    st.image(str(trajectory_evaluations_trajectories), caption="Trajectory Evaluation")

                # --- Column 3: Output Video ---
                with col3:
                    st.subheader("3. AV Decision Video")
                    st.write(
                        "A video showing how the AV makes decisions that respect human reasons according to the weights you set")
                    # Use st.video for the video
                    st.video(str(output_video_path))

            else:
                st.error("Video generation failed. Make sure the simulation produced frames.")
    except FileNotFoundError:
        st.error(f"The directory itself does not exist: {results_folder}")
    # --- END OF DIAGNOSTIC BLOCK ---


# # --- Default Outputs ---
# st.video(str(video_path))
# st.image(str(spatial_image_path), caption="Spatial Trajectory Evaluation")
# st.image(str(trajectory_image_path), caption="Trajectory Evaluation")

script_dir = Path(__file__).parent
grandparent_dir = script_dir.parent.parent
d_results_folder = grandparent_dir / "results"
d_output_video_path = d_results_folder / "d_output_video.mp4"
print(d_output_video_path)
d_trajectory_evaluation_spatial = d_results_folder / "d_trajectory_evaluations_spatial.png"
d_trajectory_evaluations_trajectories = d_results_folder / "d_trajectory_evaluations_trajectries.png"

video_path = os.path.join(script_dir, "d_output_video.mp4")
spatial_image_path = os.path.join(script_dir, "d_trajectory_evaluations_spatial.png")
trajectory_image_path = os.path.join(script_dir, "d_trajectory_evaluations_trajectories.png")

import streamlit as st

# st.set_page_config(layout="wide")

st.title("Towards Reasons-Responsive Decisions in Automated Vehicles")

st.write("This simulator shows how an automated vehicle (AV) can make **decisions that respect human reasons**.")

st.write("To do this, the AV must:")

st.markdown("""
1.  **Imagine options** – generate different possible trajectories.
2.  **Weigh them up** – check which option best fits human priorities (like safety, rules, and efficiency).
3.  **Commit** – follow the chosen path smoothly.
""")

st.write("Below, we walk through this process for the case of **overtaking a cyclist**. 🚴")

st.markdown("### 🛣️ Possible Trajectories")

st.write("Assume we have four possible trajectories: T1, T2, T3, and T4 as follows:")

import streamlit.components.v1 as components

# --- 1. Embedded HTML Content (SVG Graphic) ---
# Content is minimized to focus on the graphic and relies only on in-figure labels.
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom styles for SVG element colors, ensuring they look vibrant */
        .color-t1 { stroke: #e879f9; } /* Pink */
        .color-t2 { stroke: #3b82f6; } /* Blue */
        .color-t3 { stroke: #4b5563; } /* Gray */
        .color-t4 { stroke: #ef4444; } /* Red */

        /* Custom road line colors */
        .road-boundary { stroke: #1f2937; stroke-dasharray: 8 6; }
        .lane-line { stroke: #fbbf24; } /* Yellow/Orange */

        /* Text positioning for the T labels */
        .t-label { font-weight: bold; font-family: sans-serif; }
    </style>
</head>
<!-- Removed background from body to allow Streamlit's background, but adding white background to main graphic container -->
<body class="bg-transparent flex items-start justify-center font-sans p-0 m-0">
    <!-- Main graphic container with white background and minimal border/shadow -->
    <div class="w-full bg-white p-4 rounded-lg shadow-md border border-gray-200">
        <div class="w-full overflow-hidden">
            <!-- SVG Viewport: 800x350 -->
            <svg viewBox="0 0 800 350" class="w-full h-auto">

                <!-- Road Boundaries (Dashed Black Lines) -->
                <line x1="0" y1="50" x2="800" y2="50" class="road-boundary" stroke-width="3"/>
                <line x1="0" y1="300" x2="800" y2="300" class="road-boundary" stroke-width="3"/>

                <!-- Lane Lines (Solid Orange Lines) -->
                <line x1="0" y1="150" x2="800" y2="150" class="lane-line" stroke-width="4"/>
                <line x1="0" y1="170" x2="800" y2="170" class="lane-line" stroke-width="4"/>

                <!-- VEHICLES (Emojis) -->
                <!-- Cyclist (Placed at Y=130) -->
                <text x="350" y="140" font-size="30">🚴‍♂️</text>
                <!-- Car (Placed at Y=130) -->
                <text x="600" y="140" font-size="30">🚗</text>

                <!-- OVERTAKING PATHS (T4, T3, T2) -->
                <!-- T4: Large gap overtaking (Red, widest curve) -->
                <path
                    d="M 280 170 C 350 245, 530 245, 600 170"
                    class="color-t4"
                    fill="none"
                    stroke-width="5"
                    stroke-linecap="round"
                    stroke-dasharray="10 10"
                />
                <!-- T3: Medium gap overtaking (Gray) -->
                <path
                    d="M 280 170 C 350 225, 530 225, 600 170"
                    class="color-t3"
                    fill="none"
                    stroke-width="5"
                    stroke-linecap="round"
                    stroke-dasharray="10 10"
                />
                <!-- T2: Small gap overtaking (Blue, tightest curve) -->
                <path
                    d="M 280 170 C 350 205, 530 205, 600 170"
                    class="color-t2"
                    fill="none"
                    stroke-width="5"
                    stroke-linecap="round"
                    stroke-dasharray="10 10"
                />

                <!-- FOLLOWING PATH (T1) -->
                <!-- T1: Conservative following (Pink, straight dashed line) -->
                <line x1="480" y1="130" x2="580" y2="130" class="color-t1" stroke-width="5" stroke-dasharray="10 10" stroke-linecap="round"/>

                <!-- LABELS -->
                <!-- T4 Label (Red) -->
                <text x="220" y="175" class="t-label color-t4" font-size="18" fill="#ef4444">T4</text>
                <!-- T3 Label (Gray) -->
                <text x="250" y="160" class="t-label color-t3" font-size="18" fill="#4b5563">T3</text>
                <!-- T2 Label (Blue) -->
                <text x="270" y="145" class="t-label color-t2" font-size="18" fill="#3b82f6">T2</text>
                <!-- T1 Label (Pink) -->
                <text x="450" y="125" class="t-label color-t1" font-size="18" fill="#e879f9">T1</text>
            </svg>
        </div>
    </div>
</body>
</html>
"""

# 2. Embed the HTML content using components.html
# Adjusted height slightly since the legend is removed.
components.html(
    html_content,
    height=380,
    scrolling=False
)
# 3. Add the requested caption
st.caption("Figure 1: Comparison of Overtaking Trajectories (T2, T3, T4) and Conservative Following (T1).")

st.write("Each of the trajectories represent its compliance with human reasons:")

st.markdown("""
**T1 (Conservative following 🚦)**
* Fully follows traffic rules (policymaker's priority).
* Ignores cyclist comfort/safety and driver's time efficiency.
* Weights: Policymaker 100%, Cyclist 0%, Driver 0%.

**T2 (Small gap overtaking ⏱️)**
* Prioritizes driver's time efficiency.
* Still partly considers policymaker compliance, but only weakly considers cyclist safety.
* Weights: Policymaker 40%, Cyclist 20%, Driver 40%.

**T3 (Medium gap overtaking 🟢)**
* Balances policymaker compliance with cyclist safety.
* Driver's time efficiency not considered.
* Weights: Policymaker 50%, Cyclist 50%, Driver 0%.

**T4 (Large gap overtaking 🔴)**
* Fastest option, prioritizing both driver's time efficiency and cyclist safety.
* Does not consider policymaker compliance.
* Weights: Policymaker 0%, Cyclist 50%, Driver 50%.
""")

st.markdown("---")  # Creates the horizontal line

st.markdown("### ⚙️ Step 1: Generate Possible Paths")

st.write(
    "We use an algorithm that creates possible trajectories. It does this by giving weights to different human reasons considerations, such as rules, safety, and efficiency.")

st.markdown("""
* T1 is fixed as the conservative baseline (100% policy).
* T2-T4 can be adjusted by changing the preset weights below.
""")

st.markdown("#### Preset weights for T2-T4")
st.write("The bars below show the current weight mix used to generate each path. These are example presets only.")

st.markdown("""
* Move the sliders for 👮 Policy, 🚗 Time, and 🚲 Safety to set your own weights.
* The weights for each trajectory always sum to 100%.
* T1 stays fixed as the reference case (100% policy).
""")

st.markdown("👉 **Try it out:** Adjust the sliders to see how your chosen weights generate different overtaking paths.")

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- Configuration and Styles ---

# Set a wide layout
st.set_page_config(page_title="Weight Trajectory Simulator")

# Define colors and emojis
COLOR_POLICY = "#4CAF50"  # Green ⚖️
EMOJI_POLICY = "⚖️"
COLOR_TIME = "#2196F3"  # Blue ⏱️
EMOJI_TIME = "⏱️"
COLOR_SAFETY = "#F44336"  # Red 🚴
EMOJI_SAFETY = "🚴"

# Map for Altair (Not strictly needed here, but kept for consistency)
COLOR_MAP = {
    f"{EMOJI_POLICY} Policy": COLOR_POLICY,
    f"{EMOJI_TIME} Time Efficiency": COLOR_TIME,
    f"{EMOJI_SAFETY} Cyclist Safety": COLOR_SAFETY
}


def set_slider_colors(key, color, value):
    """
    Injects highly specific CSS to force a custom color on the slider track and cursor.
    """
    # 1. Slider Track Color (using linear-gradient)
    track_css = f'''
    <style> 
        /* Target the specific slider's track based on key */
        div[data-testid*="{key}"] div.stSlider > div[data-baseweb = "slider"] > div > div {{
            background: linear-gradient(to right, {color} 0%,  
                                                {color} {value * 100}%, 
                                                rgba(151, 166, 195, 0.25) {value * 100}%, 
                                                rgba(151, 166, 195, 0.25) 100%) !important;
        }} 
    </style>'''
    st.markdown(track_css, unsafe_allow_html=True)

    # 2. Slider Cursor Color
    cursor_css = f'''
    <style> 
        /* Target the specific slider's cursor based on key */
        div[data-testid*="{key}"] div.stSlider > div[data-baseweb="slider"] div[data-baseweb="slider-handle"]{{
            background-color: {color} !important; 
            border-color: {color} !important;
            box-shadow: {color}40 0px 0px 0px 0.2rem !important;
        }} 
    </style>'''
    st.markdown(cursor_css, unsafe_allow_html=True)

    # 3. Fix for the slider's background color on the left side (if it's white/light)
    fill_css = f'''
    <style>
        div[data-testid*="{key}"] div.stSlider > div[data-baseweb="slider"] div[style*="background-color: rgb(255, 127, 80)"] {{
            background-color: {color} !important;
        }}
    </style>
    '''
    st.markdown(fill_css, unsafe_allow_html=True)


# --- Helper Function for Stacked Bar (Native Altair) ---
def stacked_weight_bar_native(xxx, yyy, zzz, is_valid):
    """
    Creates a single, thick, stacked bar using native Streamlit (Altair).
    """

    # 1. Prepare data
    data = pd.DataFrame({
        'Category': [f"{EMOJI_POLICY} Policy", f"{EMOJI_TIME} Time Efficiency", f"{EMOJI_SAFETY} Cyclist Safety"],
        'Weight': [xxx, yyy, zzz],
    })

    # 2. Determine Opacity/Color for "Grey Out"
    opacity_color_map = {
        f"{EMOJI_POLICY} Policy": COLOR_POLICY if is_valid else "#808080",
        f"{EMOJI_TIME} Time Efficiency": COLOR_TIME if is_valid else "#808080",
        f"{EMOJI_SAFETY} Cyclist Safety": COLOR_SAFETY if is_valid else "#808080"
    }

    # --- Background Bar (Full Width) ---
    background_bar = alt.Chart(pd.DataFrame({'x': [0], 'y': [1]})).mark_bar(
        color='#333333',
        height=60
    ).encode(
        x=alt.X('x', scale=alt.Scale(domain=[0, 1]), axis=None),
        x2='y',
        y=alt.Y('constant:N', title=None, axis=None)
    ).properties(
        width='container'
    )

    # --- Actual Stacked Bar (Foreground) ---
    stacked_chart = alt.Chart(data).mark_bar(height=60).encode(
        x=alt.X('Weight', axis=None, stack="normalize"),
        y=alt.Y('constant:N', title=None, axis=None),

        color=alt.Color('Category',
                        scale=alt.Scale(domain=list(opacity_color_map.keys()),
                                        range=list(opacity_color_map.values())),
                        legend=alt.Legend(title="Weights",
                                          orient="bottom",
                                          columns=3,
                                          labelFontSize=14,
                                          titlePadding=0,
                                          titleOrient='top')),
        tooltip=['Category', alt.Tooltip('Weight', format='.1%')]
    ).properties(
        width='container'
    )

    # Combine charts and set total height
    final_chart = alt.layer(background_bar, stacked_chart).resolve_scale(
        y='independent'
    ).properties(height=100)

    st.altair_chart(final_chart, use_container_width=True)


# --- Main Slider Function (Updated to check for run button control) ---
def create_weight_sliders(
        full_title,
        initial_xxx,
        initial_yyy,
        initial_zzz,
        preset_policy_label,
        preset_driver_label,
        preset_cyclist_label,
        key_suffix="standard",  # New parameter for key suffix
        button_placeholder=None,  # New parameter to pass the button container
):
    """
    Creates the sliders, validation, stacked animated bar, and controls the run button state.

    Returns:
        (xxx, yyy, zzz, is_valid)
    """

    # Ensure a unique key prefix based on the suffix
    key_prefix = key_suffix.replace(" ", "_").replace(":", "_")

    with st.container():
        st.subheader(full_title)

        # Only show preset weights if they are provided
        if preset_policy_label is not None:
            st.write(
                f"**Preset Weights**: Policy: {preset_policy_label}%, Time: {preset_driver_label}%, Safety: {preset_cyclist_label}%")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                f"Set your weights below:"
            )

            # Policy Slider (yyy)
            policy_key = f"{key_prefix}_policy"
            xxx = st.slider(
                f"{EMOJI_POLICY} Policy",
                min_value=0.0,
                max_value=1.0,
                value=initial_yyy,
                step=0.01,
                key=policy_key
            )
            set_slider_colors(policy_key, COLOR_POLICY, xxx)

            # Time Efficiency Slider (xxx)
            driver_key = f"{key_prefix}_driver"
            yyy = st.slider(
                f"{EMOJI_TIME} Time Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=initial_xxx,
                step=0.01,
                key=driver_key
            )
            set_slider_colors(driver_key, COLOR_TIME, yyy)

            # Cyclist Safety Slider (zzz)
            cyclist_key = f"{key_prefix}_cyclist"
            zzz = st.slider(
                f"{EMOJI_SAFETY} Cyclist Safety",
                min_value=0.0,
                max_value=1.0,
                value=initial_zzz,
                step=0.01,
                key=cyclist_key
            )
            set_slider_colors(cyclist_key, COLOR_SAFETY, zzz)

        total_sum = xxx + yyy + zzz
        is_valid = np.isclose(total_sum, 1.0)

        with col2:
            st.markdown("### Current Weight Distribution")

            # Force a break line for separation
            st.markdown("<br>", unsafe_allow_html=True)

            # Display the Stacked Bar using native Altair
            stacked_weight_bar_native(xxx, yyy, zzz, is_valid)

            # Display the validation message
            if not is_valid:
                st.error("The weights must sum to 1.0 (currently sum to: {:.2f})".format(total_sum))
            else:
                st.success("Weights sum to 1.0! Policy: {:.2f}, Time: {:.2f}, Safety: {:.2f}".format(yyy, xxx, zzz))

    # Control the run button in the external placeholder container
    if button_placeholder:
        with button_placeholder:
            # The button is disabled if ANY weight set is invalid (i.e., this one)
            st.button("Run Simulation", disabled=not is_valid, type="primary")

    return xxx, yyy, zzz, is_valid


# =========================================================================
# --- MAIN APPLICATION LAYOUT ---
# =========================================================================

# --- Section 1: Original Four Trajectories ---
st.header("Step 1: Explore Preset Trajectories")

# Use a custom key_suffix for the original four sets
xxx0, yyy0, zzz0, is_evaluator_valid0 = create_weight_sliders("Trajectory 1", 0.4, 0.4, 0.2, 50, 0, 50, key_suffix="T2")
st.divider()
xxx1, yyy1, zzz1, is_evaluator_valid1 = create_weight_sliders("Trajectory 2", 0.5, 0.0, 0.5, 0, 50, 50, key_suffix="T3")
st.divider()
xxx2, yyy2, zzz2, is_evaluator_valid2 = create_weight_sliders("Trajectory 3", 0.0, 0.5, 0.5, 40, 40, 20,
                                                              key_suffix="T4")
st.divider()
xxx3, yyy3, zzz3, is_evaluator_valid3 = create_weight_sliders("Trajectory 4", 1.0, 0.0, 0.0, 100, 0, 0, key_suffix="T1")
st.divider()

# --- Section 2: Set Your Own Weights (The new set) ---
st.markdown("### ⚖️ Step 2: Choose Evaluator Weights")

st.write(
    "After generating the possible trajectories, we now **evaluate which one best matches the evaluator weights you set**.")

st.write("Use the sliders to decide how much priority to give to each reason:")

st.markdown("""
* 🚦 Policymaker (rules)
* 🚗 Driver (time efficiency)
* 🚲 Cyclist (safety)
""")

st.write(
    "For example, if you set 40% for policymaker, 30% for driver, and 30% for cyclist, the system will choose the path that is **most aligned** with that mix.")

st.markdown(
    "👉 **Try it out:** Adjust the weights according to your own preferences and see which trajectory gets selected.")

# This set needs to return its validity flag to control the button
# Use a distinct key_suffix for this set: "evaluator"
weight_policy, weight_driver, weight_cyclist, is_evaluator_valid = create_weight_sliders(
    full_title="Set your own weights",
    initial_xxx=0.33,
    initial_yyy=0.33,
    initial_zzz=0.34,
    preset_policy_label=None,  # Set to None to hide the preset line
    preset_driver_label=None,
    preset_cyclist_label=None,
    key_suffix="evaluator"
)

st.divider()

# --- Section 3: Commit and Run the Simulation (The Button) ---

# Create a container specifically for the button
button_container = st.container()

st.markdown("### ▶️ Step 3: Commit and Run the Simulation")

st.write(
    "When you click **Run Simulation**, the algorithm generates three trajectories based on the weights you set. It then identifies the trajectory that best matches those weights. A model predictive control (MPC) module is used to track the selected trajectory.")

st.write("The output is a video showing how the AV makes decisions that reflect your chosen priorities.")

st.markdown("👉 **Try it out:** Run the simulation and watch how your chosen priorities shape the AV's behavior.")

# Place the button inside the designated container.
# The disabled state is controlled by the logic below.
# with button_container:
#     # Use the validity flag from the 'Set your own weights' section to disable the button.
#     # The button is ONLY enabled if the user-defined weights sum to 1.0.
#     st.button(
#         "Run Simulation",
#         disabled=not is_evaluator_valid,
#         type="primary",
#         key="main_run_button"
#     )

# --- Parameter Inputs (new section) ---
st.subheader("⚙ Simulation Parameters ")
with st.expander("Expand to edit simulation parameters"):
    # ScenarioParameters
    st.markdown("**Scenario Parameters**")
    scenario_dt = st.number_input("DT (Time step)", value=0.1, step=0.01)
    scenario_centerline_location = st.number_input("CENTERLINE_LOCATION", value=0.0, step=0.1)
    scenario_length = st.number_input("LENGTH (Length of scenario)", value=44.0, step=1.0)
    scenario_width_road = st.number_input("WIDTH_ROAD", value=4.0, step=0.1)
    scenario_x_loc_goal = st.number_input("X_LOC_GOAL", value=2.0, step=0.1)
    scenario_y_loc_goal = st.number_input("Y_LOC_GOAL", value=22.0, step=1.0)
    scenario_x_loc_ego = st.number_input("X_LOC_EGO", value=2.0, step=0.1)
    scenario_x_loc_cyclist_buffer = st.number_input("X_LOC_CYCLIST_BUFFER", value=1.6, step=0.1)
    scenario_y_loc_cyclist_buffer = st.number_input("Y_LOC_CYCLIST_BUFFER", value=9.7, step=0.1)

    st.markdown("---")

    # ReasonParameters
    st.markdown("**Reason Parameters**")
    reasons_threshold = st.number_input("REASONS_THRESHOLD", value=0.7, step=0.1)
    st.markdown("---")

    # MPCParameters
    st.markdown("**MPC Parameters**")
    mpc_time_horizon = st.number_input("TIME_HORIZON", value=7.0, step=0.5)
    mpc_frame_window = st.number_input("FRAME_WINDOW", value=10, step=1)
    mpc_max_speed_freeway = st.number_input("MAX_SPEED_FREEWAY", value=30 / 3.6, step=1.0)
    st.markdown("---")

    # DriverParameters
    st.markdown("**Driver Parameters**")
    driver_distance_ref = st.number_input("DRIVER DISTANCE_REF", value=10.0, step=0.5)
    driver_distance_buffer = st.number_input("DRIVER DISTANCE_BUFFER", value=2.0, step=0.5)
    driver_time_threshold = st.number_input("DRIVER TIME_THRESHOLD", value=8.0, step=0.5)
    st.markdown("---")

    # CyclistParameters
    st.markdown("**Cyclist Parameters**")
    cyclist_distance_ref = st.number_input("CYCLIST DISTANCE_REF", value=8.0, step=0.5)
    cyclist_distance_buffer = st.number_input("CYCLIST DISTANCE_BUFFER", value=2.0, step=0.5)
    cyclist_time_threshold = st.number_input("CYCLIST TIME_THRESHOLD", value=5.0, step=0.5)
    cyclist_speed = st.number_input("CYCLIST SPEED (km/h)", value=5 / 3.6, step=0.5)
    st.markdown("---")

    with st.container():
        st.subheader("ideal weight set")
        ideal_weight_cyclist = st.slider("ideal_weight_cyclist", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
        ideal_weight_driver = st.slider("ideal_weight_driver", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
        ideal_weight_policymaker = st.slider("ideal_weight_policymaker", min_value=0.0, max_value=1.0, value=0.34,
                                             step=0.01)

# st.button(
#         "Run Simulation",
#         disabled=not is_evaluator_valid,
#         type="primary",
#         key="main_run_button"
#     )
if st.button("Run Simulation",
             disabled=not (
                     is_evaluator_valid and is_evaluator_valid0 and is_evaluator_valid1 and is_evaluator_valid2 and is_evaluator_valid3),
             type="primary"):
    st.info("Running simulation...")

    # Replace the video with an embedded Dino game clone
    dino_html = """
    <iframe src="https://chromedino.com/" 
            width="100%" height="500" 
            style="border:none;">
    </iframe>
    """
    with st.expander("See Default Outputs"):

        # Create three columns with equal width (1, 1, 1) inside the expander
        col1, col2, col3 = st.columns(3)

        # --- Column 1: Spatial Trajectory Evaluation ---
        with col1:
            st.subheader("1. Generated Trajectories from Step 1")
            # Display the spatial image
            st.image(str(spatial_image_path), caption="Spatial Trajectory Evaluation")

        # --- Column 2: Trajectory Evaluation Results ---
        with col2:
            st.subheader("2. Evaluation Results of Three Trajectories from Step 2")
            # Display the trajectory image
            st.image(str(trajectory_image_path), caption="Trajectory Evaluation")

        # --- Column 3: Output Video ---
        with col3:
            st.subheader("3. AV Decision Video")
            st.write(
                "A video showing how the AV makes decisions that respect human reasons according to the weights you set")
            # Display the video
            st.video(str(video_path))

    components.html(dino_html, height=500)
    ignoree = Path(__file__).resolve().parent
    print(ignoree)

    # Normalize 1st set
    total1 = weight_policy + weight_driver + weight_cyclist
    if total1 == 0:
        norm_policy = norm_driver = norm_cyclist = round(1 / 3, 1)
    else:
        norm_policy = round(weight_policy / total1, 1)
        norm_driver = round(weight_driver / total1, 1)
        norm_cyclist = round(weight_cyclist / total1, 1)

    # # Normalize 2nd set
    # total2 = weight_xxx + weight_yyy + weight_zzz
    # if total2 == 0:
    #     xxx = yyy = zzz = round(1/3, 1)
    # else:
    #     xxx = round(weight_xxx / total2, 1)
    #     yyy = round(weight_yyy / total2, 1)
    #     zzz = round(weight_zzz / total2, 1)

    # Normalize ideal set
    total3 = ideal_weight_cyclist + ideal_weight_driver + ideal_weight_policymaker
    if total3 == 0:
        ideal_weight_cyclist = ideal_weight_driver = ideal_weight_policymaker = round(1 / 3, 1)
    else:
        ideal_weight_cyclist = round(ideal_weight_cyclist / total3, 1)
        ideal_weight_driver = round(ideal_weight_driver / total3, 1)
        ideal_weight_policymaker = round(ideal_weight_policymaker / total3, 1)

    # Gather all parameters into a dictionary
    params_to_update = {
        'scenario_dt': scenario_dt,
        'scenario_centerline_location': scenario_centerline_location,
        'scenario_length': scenario_length,
        'scenario_width_road': scenario_width_road,
        'scenario_x_loc_goal': scenario_x_loc_goal,
        'scenario_y_loc_goal': scenario_y_loc_goal,
        'scenario_x_loc_ego': scenario_x_loc_ego,
        'scenario_x_loc_cyclist_buffer': scenario_x_loc_cyclist_buffer,
        'scenario_y_loc_cyclist_buffer': scenario_y_loc_cyclist_buffer,
        'reasons_threshold': reasons_threshold,
        'mpc_time_horizon': mpc_time_horizon,
        'mpc_frame_window': mpc_frame_window,
        'mpc_max_speed_freeway': mpc_max_speed_freeway,
        'driver_distance_ref': driver_distance_ref,
        'driver_distance_buffer': driver_distance_buffer,
        'driver_time_threshold': driver_time_threshold,
        'cyclist_distance_ref': cyclist_distance_ref,
        'cyclist_distance_buffer': cyclist_distance_buffer,
        'cyclist_time_threshold': cyclist_time_threshold,
        'cyclist_speed': cyclist_speed,
    }

    # Update parameters file before running the simulation
    update_parameters_file(params_to_update)
    st.success(f"Parameters in '{PARAMETERS_PATH.name}' updated successfully.")

    import lib.parameters

    importlib.reload(lib.parameters)
    from lib.parameters import CyclistParameters, DriverParameters, ScenarioParameters, MPCParameters, ReasonParameters

    run_simulation(norm_policy, norm_driver, norm_cyclist, ideal_weight_cyclist, ideal_weight_driver,
                   ideal_weight_policymaker, xxx0, xxx1, xxx2, xxx3, yyy0, yyy1, yyy2, yyy3, zzz0, zzz1, zzz2, zzz3)
