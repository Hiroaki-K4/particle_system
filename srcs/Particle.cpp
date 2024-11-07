#include "Particle.hpp"

// Particle::Particle(int particle_num) { initialize_position_randomly(particle_num); }
Particle::Particle(int particle_num, float aspect_ratio) {
    gravity_pos = glm::vec2(0.0f, 0.0f);
    initialize_position(particle_num, aspect_ratio);
}

std::vector<glm::vec2> Particle::get_position() { return this->position; }

void Particle::initialize_position_randomly(int particle_num) {
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < particle_num; i++) {
        glm::vec2 pos;
        pos.x = dis(gen);
        pos.y = dis(gen);
        this->position.push_back(pos);

        glm::vec2 dir;
        dir.x = dis(gen);
        dir.y = dis(gen);
        dir = glm::normalize(dir);
        dir.x = dir.x * 0.01;
        dir.y = dir.y * 0.01;
        this->dir_vec.push_back(dir);
    }
}

void Particle::initialize_position(int particle_num, float aspect_ratio) {
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, 0.2f);
    for (int i = 0; i < particle_num; i++) {
        glm::vec2 pos;
        float angle = dis(gen);
        float radius = radius_dis(gen);
        pos.x = cos(angle) * aspect_ratio * radius;
        pos.y = sin(angle) * radius;
        this->position.push_back(pos);

        glm::vec2 dir;
        dir.x = dis(gen);
        dir.y = dis(gen);
        dir = glm::normalize(dir);
        dir.x = dir.x * 0.01;
        dir.y = dir.y * 0.01;
        this->dir_vec.push_back(dir);
    }
}

glm::vec2 Particle::calculate_reflection_vector(const glm::vec2 &I, const glm::vec2 &n) {
    glm::vec2 normalized_n = glm::normalize(n);
    float dot_product = glm::dot(I, normalized_n);
    glm::vec2 reflection = I - 2 * dot_product * normalized_n;

    return reflection;
}

// void update_

void Particle::update_position_according_to_direction() {
    for (std::size_t i = 0; i < this->position.size(); ++i) {
        this->position[i].x =
            std::min(std::max(this->position[i].x + this->dir_vec[i].x, -1.0f), 1.0f);
        this->position[i].y =
            std::min(std::max(this->position[i].y + this->dir_vec[i].y, -1.0f), 1.0f);

        if (((this->position[i].x == 1.0f) || (this->position[i].x == -1.0f)) &&
            ((this->position[i].y == 1.0f) || (this->position[i].y == -1.0f))) {
            this->dir_vec[i].x = this->dir_vec[i].x * (-1);
            this->dir_vec[i].y = this->dir_vec[i].y * (-1);
        } else if (this->position[i].x == 1.0f) {
            glm::vec2 n(-1.0f, 0.0f);
            glm::vec2 reflect = calculate_reflection_vector(this->dir_vec[i], n);
            this->dir_vec[i] = reflect;
        } else if (this->position[i].x == -1.0f) {
            glm::vec2 n(1.0f, 0.0f);
            glm::vec2 reflect = calculate_reflection_vector(this->dir_vec[i], n);
            this->dir_vec[i] = reflect;
        } else if (this->position[i].y == 1.0f) {
            glm::vec2 n(0.0f, -1.0f);
            glm::vec2 reflect = calculate_reflection_vector(this->dir_vec[i], n);
            this->dir_vec[i] = reflect;
        } else if (this->position[i].y == -1.0f) {
            glm::vec2 n(0.0f, 1.0f);
            glm::vec2 reflect = calculate_reflection_vector(this->dir_vec[i], n);
            this->dir_vec[i] = reflect;
        }
    }
}
