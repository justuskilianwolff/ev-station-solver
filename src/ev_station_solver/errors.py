class IntegerInfeasible(Exception):
    def __init__(self):
        message = (
            "Model is infeasible. Please add more initial locations and/or increase the fixed number of chargers (if fixed)."
        )
        super().__init__(message)
