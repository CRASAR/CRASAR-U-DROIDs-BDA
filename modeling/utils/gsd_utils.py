import numpy as np

def get_gsd_steps(min_gsd_x, min_gsd_y, max_gsd_x, max_gsd_y, multiscale_steps, multiscale_step_interval):
    if multiscale_step_interval.lower() == "logspace":
        multiscale_steps_gsd_x = np.logspace(np.log10(min_gsd_x), np.log10(max_gsd_x), num=multiscale_steps)
        multiscale_steps_gsd_y = np.logspace(np.log10(min_gsd_y), np.log10(max_gsd_y), num=multiscale_steps)
    elif multiscale_step_interval.lower() == "linspace":
        multiscale_steps_gsd_x = np.linspace(min_gsd_x, max_gsd_x, num=multiscale_steps)
        multiscale_steps_gsd_y = np.linspace(min_gsd_y, max_gsd_y, num=multiscale_steps)
    else:
        raise ValueError("Unknown multiscale_step_interval " + str(multiscale_step_interval) + " valid options are logspace and linspace.")
    return multiscale_steps_gsd_x, multiscale_steps_gsd_y