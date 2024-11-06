#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "Shader.hpp"


void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
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

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Initialize window
    glViewport(0, 0, window_w, window_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    std::vector<float> circle_vertices = generate_circle_vertices(0.0f, 0.0f, 0.05f, 100, float(window_h)/float(window_w));

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    Shader particle_shader("../shaders/particle.vs", "../shaders/particle.fs");

    // Generate a list of 100 quad  locations/translation-vectors
    glm::vec2 translations[100];
    int index = 0;
    float offset = 0.1f;
    for (int y = -10; y < 10; y += 2) {
        for (int x = -10; x < 10; x += 2) {
            glm::vec2 translation;
            translation.x = (float)x / 10.0f + offset;
            translation.y = (float)y / 10.0f + offset;
            translations[index++] = translation;
        }
    }

    // Store instance data in an array buffer
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, circle_vertices.size() * sizeof(float), circle_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // set instance data
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(1, 1); // tell OpenGL this is an instanced vertex attribute

    double last_time = glfwGetTime();
    int frame_num = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        translations[0].x += 0.001;
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT);

        particle_shader.use();
        glBindVertexArray(VAO);
        // glDrawArraysInstanced(GL_TRIANGLES, 0, circle_vertices.size(), 100);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, circle_vertices.size() / 2, 100);  // Use GL_TRIANGLE_FAN
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Measure FPS
        double current_time = glfwGetTime();
        double delta = current_time - last_time;
        frame_num += 1;
        if (delta >= 1.0) {
            int fps = int(double(frame_num) / delta);
            std::stringstream ss;
            ss << window_title.c_str() << " [" << fps << " FPS]";
            glfwSetWindowTitle(window, ss.str().c_str());
            frame_num = 0;
            last_time = current_time;
        }
    }

    glfwTerminate();
    return 0;
}
