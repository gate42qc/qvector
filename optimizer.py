from scipy.optimize import minimize


class LBFGSOptimizer:
    def __init__(self):
        self.iterator = 0

    def optimize(self, cost_function, initial_params, args):
        def callback(x):
            self.iterator += 1
            print(f"Running {self.iterator}th iteration with params: {x}")

        return minimize(cost_function, initial_params, args, method='L-BFGS-B', options={'maxiter': 10, 'disp': True},
                        callback=callback)
