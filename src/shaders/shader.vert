#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj;
} ubo;

layout(location = 0) in vec3 pos;
layout(location = 1) in float size;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.modelViewProj*vec4(pos, 1.0);
    gl_PointSize = 5*size/gl_Position.w;
    fragColor = color;
}