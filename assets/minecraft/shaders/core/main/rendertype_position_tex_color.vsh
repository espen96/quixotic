#version 150

in vec3 Position;
in vec2 UV0;
in vec2 UV2;
in vec4 Color;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
out float lmx;
out float lmy;
out vec2 texCoord0;
out vec4 vertexColor;

out vec4 glpos;

void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;
    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
    texCoord0 = UV0;
    vertexColor = Color;
}
