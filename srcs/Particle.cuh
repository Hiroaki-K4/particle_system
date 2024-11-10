#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "kernel.cuh"


class Particle {
   private:
    glm::vec2 gravity_pos;
    std::vector<glm::vec2> position;
    std::vector<glm::vec2> velocity;
    std::vector<glm::vec3> color;
    float max_distance;
    // Cuda
    glm::vec2 *cu_position;
    glm::vec2 *cu_velocity;
    glm::vec3 *cu_color;
    int threads;
    int blocks;

   public:
    Particle(int particle_num, float aspect_ratio, int threads);
    ~Particle();

    std::vector<glm::vec2> get_position();
    std::vector<glm::vec3> get_color();

    void set_gravity_pos(float x, float y);

    void initialize_position(int particle_num, float aspect_ratio);

    void update_position_velocity_color(float delta_time, float aspect_ratio);
    void create_new_color(float distance, glm::vec3 &new_color);

};

#endif
