#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>
#include <vector>

class Particle {
   private:
    glm::vec2 gravity_pos;
    std::vector<glm::vec2> position;
    std::vector<glm::vec2> dir_vec;
    std::vector<glm::vec2> velocity;

   public:
    Particle(int particle_num, float aspect_ratio);
    std::vector<glm::vec2> get_position();
    void set_gravity_pos(float x, float y);

    void initialize_position_randomly(int particle_num);
    void initialize_position(int particle_num, float aspect_ratio);

    glm::vec2 calculate_reflection_vector(const glm::vec2 &I, const glm::vec2 &n);
    void update_position_according_to_direction();
    void update_position(float delta_time, float aspect_ratio);

};

#endif
