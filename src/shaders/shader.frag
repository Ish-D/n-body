#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    const float radius = 0.5;
    float alpha = 1/(2+pow(15*length(gl_PointCoord-vec2(radius, radius)), 2));

    if (length(gl_PointCoord - vec2(0.5)) > radius || alpha <= 0.10)
        discard;
    outColor = vec4(fragColor, alpha+0.5);
}