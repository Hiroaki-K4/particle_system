#include "kernel.cuh"
#include <iostream>

__global__ void update_particle_kernel(
    glm::vec2 *cu_position, glm::vec2 *cu_velocity, glm::vec3 *cu_color, glm::vec2 gravity_pos,
    float delta_time, float aspect_ratio, int num_particles, float max_distance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) {
        return;
    }

    // Update velocity and position
    glm::vec2 rescaled_pos = cu_position[idx];
    rescaled_pos.x /= aspect_ratio;
    glm::vec2 accel = gravity_pos - rescaled_pos;
    glm::vec2 upscale_accel = accel * glm::length(accel) * 10.0f;
    cu_velocity[idx].x += upscale_accel.x * delta_time;
    cu_velocity[idx].y += upscale_accel.y * delta_time;
    cu_position[idx].x += cu_velocity[idx].x * delta_time * aspect_ratio;
    cu_position[idx].y += cu_velocity[idx].y * delta_time;

    // Update color
    float new_color_val = fminf(glm::length(accel) / max_distance, 1.0f);
    cu_color[idx] = glm::vec3(1.0f - new_color_val, 0.0f, new_color_val);
}
