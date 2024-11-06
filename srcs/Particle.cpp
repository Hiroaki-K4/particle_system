#include "Particle.hpp"

Particle::Particle(int particle_num) {
    initialize_position_randomly(particle_num);
}

void Particle::initialize_position_randomly(int particle_num) {
    // float offset = 0.1f;
    // for (int y = -10; y < 10; y += 2) {
    //     for (int x = -10; x < 10; x += 2) {
    //         glm::vec2 translation;
    //         translation.x = (float)x / 10.0f + offset;
    //         translation.y = (float)y / 10.0f + offset;
    //         translations[index++] = translation;
    //     }
    // }
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::cout << "random: " << dis(gen) << std::endl;
    std::cout << "random: " << dis(gen) << std::endl;
    for (int i = 0; i < particle_num; i++) {
        glm::vec2 pos;
        pos.x = dis(gen);
        pos.y = dis(gen);
        this->position.push_back(pos);
    }
}

std::vector<glm::vec2> Particle::get_position() {
    return this->position;
}
