#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <random>
#include <iostream>


class Particle {
    private:
        std::vector<glm::vec2> position;
        std::vector<glm::vec2> dir_vec;

    public:
        Particle(int particle_num);
        void initialize_position_randomly(int particle_num);
        std::vector<glm::vec2> get_position();
};

#endif
