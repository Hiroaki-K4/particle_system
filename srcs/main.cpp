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
    vertices.push_back(0.0f);

    for (int i = 0; i <= num_segments; ++i) {
        float angle = 2.0f * M_PI * i / num_segments;
        vertices.push_back(center_x + radius * cos(angle) * aspect_ratio);
        vertices.push_back(center_y + radius * sin(angle));
        vertices.push_back(0.0f);
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

    // std::vector<float> circle_vertices = generate_circle_vertices(0.0f, 0.0f, 0.5f, 100, float(window_h)/float(window_w));

    std::vector<std::vector<float>> circles;
    float aspect_ratio = float(window_h) / float(window_w);
    float radius = 0.1f;
    int num_segments = 100;
    for (int i = 0; i < 5; ++i) {
        float x = -0.8f + i * 0.4f; // Adjust positions horizontally
        float y = 0.0f;
        circles.push_back(generate_circle_vertices(x, y, radius, num_segments, aspect_ratio));
    }

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // glBindVertexArray(VAO);
    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, circle_vertices.size() * sizeof(float), circle_vertices.data(), GL_STATIC_DRAW);

    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);

    Shader particle_shader("../shaders/particle.vs", "../shaders/particle.fs");

    double last_time = glfwGetTime();
    int frame_num = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClear(GL_COLOR_BUFFER_BIT);
        particle_shader.use();

        // Draw circle
        // glBindVertexArray(VAO);
        // glDrawArrays(GL_TRIANGLE_FAN, 0, circle_vertices.size() / 3);

        // Draw each circle
        for (const auto& circle_vertices : circles) {
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, circle_vertices.size() * sizeof(float), circle_vertices.data(), GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            glDrawArrays(GL_TRIANGLE_FAN, 0, circle_vertices.size() / 3);
        }

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
