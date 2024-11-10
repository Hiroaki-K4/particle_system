#include "Particle.hpp"

// Particle::Particle(int particle_num) { initialize_position_randomly(particle_num); }
Particle::Particle(int particle_num, float aspect_ratio) {
    this->gravity_pos = glm::vec2(0.0f, 0.0f);
    initialize_position(particle_num, aspect_ratio);
}

std::vector<glm::vec2> Particle::get_position() { return this->position; }

std::vector<glm::vec3> Particle::get_color() { return this->color; }

void Particle::set_gravity_pos(float x, float y) {
    this->gravity_pos.x = x;
    this->gravity_pos.y = y;
}

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
    std::uniform_real_distribution<float> radius_dis(0.0f, 0.4f);
    for (int i = 0; i < particle_num; i++) {
        glm::vec2 pos;
        float angle = dis(gen);
        float radius = radius_dis(gen);
        pos.x = cos(angle) * aspect_ratio * radius;
        pos.y = sin(angle) * radius;
        this->position.push_back(pos);

        glm::vec3 initial_color(0.0f, 0.0f, 0.0f);
        create_new_color(glm::length(this->gravity_pos - pos), initial_color);
        this->color.push_back(initial_color);
    }

    std::vector<glm::vec2> velo(particle_num, glm::vec2(0.0f, 0.0f));
    this->velocity = velo;
}

glm::vec2 Particle::calculate_reflection_vector(const glm::vec2 &I, const glm::vec2 &n) {
    glm::vec2 normalized_n = glm::normalize(n);
    float dot_product = glm::dot(I, normalized_n);
    glm::vec2 reflection = I - 2 * dot_product * normalized_n;

    return reflection;
}

void Particle::update_position_and_color(float delta_time, float aspect_ratio) {
    glm::vec3 new_color(0.0f, 0.0f, 0.0f);
    for (std::size_t i = 0; i < this->position.size(); ++i) {
        glm::vec2 rescaled_pos = this->position[i];
        rescaled_pos.x /= aspect_ratio;
        glm::vec2 accel = this->gravity_pos - rescaled_pos;
        glm::vec2 upscale_accel = accel * glm::length(accel) * 10.0f;

        this->velocity[i].x += upscale_accel.x * delta_time;
        this->velocity[i].y += upscale_accel.y * delta_time;
        this->position[i].x += this->velocity[i].x * delta_time * aspect_ratio;
        this->position[i].y += this->velocity[i].y * delta_time;

        create_new_color(glm::length(accel), new_color);
        this->color[i] = new_color;
    }
}

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

void Particle::create_new_color(float distance, glm::vec3 &new_color) {
    float max_distance = sqrt(2);
    float new_color_val = std::min(distance / max_distance, 1.0f);
    new_color[0] = 1.0f - new_color_val;
    new_color[2] = new_color_val;
}
