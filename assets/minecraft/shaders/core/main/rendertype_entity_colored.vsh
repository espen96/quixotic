#version 150

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in vec2 UV1;
in vec2 UV2;
in vec3 Normal;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

out vec4 vertexColor;
out vec2 texCoord0;
out float lmx;
out float lmy;
//out vec4 normal;

out vec4 glpos;

void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;

    vertexColor = Color;
    texCoord0 = UV0;

    //normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);
}
