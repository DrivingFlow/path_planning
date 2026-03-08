# Unified model input for both map_frame and agent_centered models.
# mode: 0 = map_frame (occ_0..occ_4 each 1216*1216; deltas/motion empty), 1 = agent_centered (10 grids 201*201 + motion).
# occ_*: float32, 0.0 or 1.0 (occupancy).
# delta_*: float32, -1.0 (newly free), 0.0 (no change), or 1.0 (newly occupied); delta_0 is zeros. Matches x_grids[:,1,:,:] = occ[1:]-occ[:-1].
# Row-major flattened.
std_msgs/Header header
uint8 mode
float32[] occ_0
float32[] occ_1
float32[] occ_2
float32[] occ_3
float32[] occ_4
float32[] delta_0
float32[] delta_1
float32[] delta_2
float32[] delta_3
float32[] delta_4
float32[] motion_forward_speed
float32[] motion_yaw_rate
