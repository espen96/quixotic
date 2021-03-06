#version 150

in vec4 Position;

uniform mat4 ProjMat;

out vec2 texCoord;

void main() {
    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * clamp(1.0 + 0.01, 0.0, 1.0) * 2.0 - 1.0;

    texCoord = outPos.xy * 0.5 + 0.5;
}
