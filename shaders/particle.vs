#version 400 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aOffset;
layout (location = 2) in vec3 aColor;

out vec3 fragColor;

void main() {
    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);
    fragColor = aColor;
}
