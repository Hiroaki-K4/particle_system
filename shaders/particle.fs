#version 400 core
in vec3 fragColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
    // FragColor = vec4(fragColor, 1.0);
}
