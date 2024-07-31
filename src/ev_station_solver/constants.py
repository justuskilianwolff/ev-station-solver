MOPTA_CONSTANTS = {
    "mu_charging": 0.4203,  # mean of vehicles charging
    "max_size": 8,  # 2 * 8 vehicles
    # ranges
    "mu_range": 100,  # mean range of vehicles
    "sigma_range": 50,  # std of range of vehicles
    "lb_range": 20,  # lower bound of range of vehicles
    "ub_range": 250,  # upper bound of range of vehicles
    # charging probability
    "lambda_charge": 0.012,
    # problem set up
    "station_ub": 8,
    "service_level": 0.95,  # service rate
    "build_cost": 5000,
    "maintenance_cost": 500,
    "drive_cost": 0.041,
    "charge_cost": 0.0388,
    "min_distance": 0.5,  # min distance in which a new charger is immediately built
    "counting_radius": 10,  # radius for counting vehicles and charging locations
}
