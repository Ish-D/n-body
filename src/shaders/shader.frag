#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    const float radius = 0.5;
    if (length(gl_PointCoord - vec2(0.5)) > radius)
        discard;
    outColor = vec4(fragColor, 1.0);
}