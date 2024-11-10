#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "Particle.cuh"
#include "Shader.hpp"


Particle* particle_system;
bool is_update_gravity_point = false;


void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        if (is_update_gravity_point) {
            is_update_gravity_point = false;
        } else {
            is_update_gravity_point = true;
        }
    }
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (is_update_gravity_point) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        // Normalize mouse position to range [-1, 1]
        float normalized_x = static_cast<float>(xpos) / width * 2.0f - 1.0f;
        float normalized_y = 1.0f - static_cast<float>(ypos) / height * 2.0f;

        particle_system->set_gravity_pos(normalized_x, normalized_y);
    }
}

std::vector<float> generate_circle_vertices(float center_x, float center_y, float radius,
                                            int num_segments, float aspect_ratio) {
    std::vector<float> vertices;
    vertices.push_back(center_x);
    vertices.push_back(center_y);

    for (int i = 0; i <= num_segments; ++i) {
        float angle = 2.0f * M_PI * i / num_segments;
        vertices.push_back(center_x + radius * cos(angle) * aspect_ratio);
        vertices.push_back(center_y + radius * sin(angle));
    }
    return vertices;
}

int main() {
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int window_w = 1920;
    int window_h = 1080;
    std::string window_title = "Partical System";
    GLFWwindow* window = glfwCreateWindow(window_w, window_h, window_title.c_str(), NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize window
    glViewport(0, 0, window_w, window_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    float aspect_ratio = float(window_h)/float(window_w);
    std::vector<float> circle_vertices = generate_circle_vertices(0.0f, 0.0f, 0.0005f, 10, aspect_ratio);

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    Shader particle_shader("../shaders/particle.vs", "../shaders/particle.fs");

    int particle_num = 1000000;
    Particle particle(particle_num, aspect_ratio);
    particle_system = &particle;

    // Store instance data in an array buffer
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * particle_num, particle.get_position().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    unsigned int instance_color_VBO;
    glGenBuffers(1, &instance_color_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, instance_color_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * particle_num, particle.get_color().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, circle_vertices.size() * sizeof(float), circle_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // set instance data(position)
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(1, 1); // tell OpenGL this is an instanced vertex attribute

    // set instance data(color)
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, instance_color_VBO);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(2, 1); // tell OpenGL this is an instanced vertex attribute

    double last_time = glfwGetTime();
    double fps_last_time = glfwGetTime();
    int frame_num = 0;
    while (!glfwWindowShouldClose(window)) {

        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT);

        double current_time = glfwGetTime();
        double delta = current_time - last_time;
        particle.update_position_and_color(delta, aspect_ratio);
        last_time = glfwGetTime();
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * particle_num, particle.get_position().data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, instance_color_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particle_num, particle.get_color().data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        particle_shader.use();
        glBindVertexArray(VAO);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, circle_vertices.size() / 2, particle_num);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Measure FPS
        double fps_current_time = glfwGetTime();
        double fps_delta = fps_current_time - fps_last_time;
        frame_num += 1;
        if (fps_delta >= 1.0) {
            int fps = int(double(frame_num) / fps_delta);
            std::stringstream ss;
            ss << window_title.c_str() << " [" << fps << " FPS]";
            glfwSetWindowTitle(window, ss.str().c_str());
            frame_num = 0;
            fps_last_time = fps_current_time;
        }
    }

    glfwTerminate();
    return 0;
}
