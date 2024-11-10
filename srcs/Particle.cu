#include "Particle.cuh"

Particle::Particle(int particle_num, float aspect_ratio, int threads, float gravity_strength) {
    this->gravity_pos = glm::vec2(0.0f, 0.0f);
    this->max_distance = sqrt(2);
    this->gravity_strength = gravity_strength;
    this->threads = threads;
    initialize_position(particle_num, aspect_ratio);
    this->blocks = (this->position.size() + threads - 1) / threads;

    // Allocate device memory
    cudaMalloc(&this->cu_position, particle_num * sizeof(glm::vec2));
    cudaMalloc(&this->cu_velocity, particle_num * sizeof(glm::vec2));
    cudaMalloc(&this->cu_color, particle_num * sizeof(glm::vec3));

    cudaMemcpy(this->cu_position, this->position.data(), particle_num * sizeof(glm::vec2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->cu_velocity, this->velocity.data(), particle_num * sizeof(glm::vec2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->cu_color, this->color.data(), particle_num * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
}

Particle::~Particle() {
    cudaFree(this->cu_position);
    cudaFree(this->cu_velocity);
    cudaFree(this->cu_color);
}

std::vector<glm::vec2> Particle::get_position() { return this->position; }

std::vector<glm::vec3> Particle::get_color() { return this->color; }

void Particle::set_gravity_pos(float x, float y) {
    this->gravity_pos.x = x;
    this->gravity_pos.y = y;
}

void Particle::initialize_position(int particle_num, float aspect_ratio) {
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    // Define a distribution between -1 and 1
    std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, 0.4f);
    glm::vec3 initial_color;
    for (int i = 0; i < particle_num; i++) {
        glm::vec2 pos;
        float angle = dis(gen);
        float radius = radius_dis(gen);
        pos.x = cos(angle) * aspect_ratio * radius;
        pos.y = sin(angle) * radius;
        this->position.push_back(pos);

        create_new_color(glm::length(this->gravity_pos - pos), initial_color);
        this->color.push_back(initial_color);
    }

    std::vector<glm::vec2> velo(particle_num, glm::vec2(0.0f, 0.0f));
    this->velocity = velo;
}

void Particle::update_position_velocity_color(float delta_time, float aspect_ratio) {
    update_particle_kernel<<<this->blocks, this->threads>>>(
        this->cu_position, this->cu_velocity, this->cu_color, this->gravity_pos, delta_time,
        aspect_ratio, this->position.size(), this->max_distance, this->gravity_strength);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        Particle::~Particle();
        exit(1);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(this->position.data(), this->cu_position, this->position.size() * sizeof(glm::vec2),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->color.data(), this->cu_color, this->position.size() * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
}

void Particle::create_new_color(float distance, glm::vec3 &new_color) {
    float max_distance = sqrt(2);
    float new_color_val = std::min(distance / max_distance, 1.0f);
    new_color[0] = 1.0f - new_color_val;
    new_color[2] = new_color_val;
}
