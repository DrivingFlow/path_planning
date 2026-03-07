# Array of occupancy grids (e.g. 5 input frames or N predicted frames).
# For agent-centered model input: send 10 grids (occ_0, delta_0, occ_1, delta_1, ..., occ_4, delta_4)
# with motion_forward_speed and motion_yaw_rate (length 5; first element 0).
std_msgs/Header header
nav_msgs/OccupancyGrid[] grids
# Motion context (agent-centered only): forward speed (m/s) and yaw rate (rad/s) per input frame. Length N_INPUT; first row zeros.
float32[] motion_forward_speed
float32[] motion_yaw_rate
