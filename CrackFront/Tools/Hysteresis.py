import numpy as np

def direction_change_index(displacements):
    """
    returns the index of the maximal displcacements
    """
    for i in range(0,len(displacements)):
        if displacements[i] > displacements[i+1]:
            return i

def split_data_forward_backward(displacements, data):
    """
    if the forward data is incomplete (becaue the jump out distance was bigger then the starting distance),
    data is padded with zeros
    """
    TOL=1e-14
    i_dirchange =  direction_change_index(displacements)

    forward_slice = slice(None, i_dirchange+1)
    backward_slice = slice(i_dirchange, None)

    backward_displacements = displacements[backward_slice]

    backward_data = []
    for d in data:
        backward_data.append(d[backward_slice])

    forward_displacements = backward_displacements
    # assert the displacement spacing forward and backward was the same
    real_forward_disp = displacements[forward_slice] # real data
    len_real_forward_data = len(real_forward_disp)
    lendiff = len_real_forward_data - len(forward_displacements)
    assert ((forward_displacements[:len_real_forward_data] - real_forward_disp[:lendiff-1 if lendiff > 0 else None:-1]) < TOL).all()

    forward_data = []
    for d in data:
        d_forward = np.zeros_like(forward_displacements, dtype=d.dtype)
        d_forward[:len_real_forward_data] = d[forward_slice][:lendiff-1 if lendiff > 0 else None:-1]
        forward_data.append(d_forward)

    return forward_displacements, forward_data, backward_data

def split_data_forward_backward_demo(displacement, force):
    import matplotlib.pyplot as plt
    forward_displacement, \
    (forward_force,), \
    (backward_force,) \
      = split_data_forward_backward(displacement, [force])

    fig, (ax_fd, axdiff) = plt.subplots(2,1, sharex=True)

    ax_fd.plot(forward_displacement, forward_force)
    ax_fd.plot(forward_displacement, backward_force)
    axdiff.plot(forward_displacement, backward_force - forward_force)

    return fig, (ax_fd, axdiff)

def compute_mean_force_diff_vs_check_size(directories):
    try:
        mean_force_diff = []
        check_size = []
        for directory in directories:
            os.chdir(basedir)
            os.chdir (directory)
            nc = NCStructuredGrid("data.nc")

            simsetup = run_path("simulation.py")

            forward_displacement, \
            (forward_force, forward_area), \
            (backward_force, backward_area) \
              = split_data_forward_backward(nc.displacement, [nc.normal_force, nc.contact_area])

            jumpin_index = len(forward_area) -1 
            while forward_area[jumpin_index] == 0:
                jumpin_index -= 1

            mean_force_diff.append(-np.mean(backward_force[:jumpin_index+1] - forward_force[:jumpin_index+1]))
            check_size.append(simsetup["pts_per_check"] * simsetup["dx"])
    finally: 
         os.chdir(basedir)
    return check_size, mean_force_diff

def find_jumpin_index(contact_area):
    """
    For simulations with contact constraint only
    """
    jumpin_index = 0
    while contact_area[jumpin_index] == 0:
        jumpin_index += 1
    return jumpin_index