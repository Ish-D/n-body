#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj;
} ubo;

layout(location = 0) in vec4 point;

layout(location = 0) out vec3 fragColor;

void main() {
    
    gl_Position = ubo.modelViewProj*vec4(point.xyz, 1.0);
    gl_PointSize = 2*point.w * (1/distance(gl_Position, vec4(point.xyz, 1.0)));
    float color = (1.0f);
    fragColor = vec3(color);
}