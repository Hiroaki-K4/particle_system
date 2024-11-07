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
    std::vector<glm::vec2> position;
    std::vector<glm::vec2> dir_vec;

   public:
    Particle(int particle_num);
    std::vector<glm::vec2> get_position();

    void initialize_position_randomly(int particle_num);

    glm::vec2 calculate_reflection_vector(const glm::vec2 &I, const glm::vec2 &n);
    void update_position_according_to_direction();

};

#endif
