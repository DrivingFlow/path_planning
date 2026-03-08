# Unified model input for both map_frame and agent_centered models.
# mode: 0 = map_frame (occ_0..occ_4 each 1216*1216 binary; deltas/motion empty), 1 = agent_centered (10 grids 201*201 + motion).
# occ_*: uint8, 0 or 1 (occupancy in [0,1]).
# delta_*: int8, -1 (newly free), 0 (no change), or 1 (newly occupied) to match x_grids[:,1,:,:] = occ[1:]-occ[:-1]; delta_0 is zeros.
# Row-major flattened.
std_msgs/Header header
uint8 mode
uint8[] occ_0
uint8[] occ_1
uint8[] occ_2
uint8[] occ_3
uint8[] occ_4
int8[] delta_0
int8[] delta_1
int8[] delta_2
int8[] delta_3
int8[] delta_4
float32[] motion_forward_speed
float32[] motion_yaw_rate
