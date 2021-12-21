#version 150

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in vec2 UV2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform mat4 TextureMat;

out float vertexDistance;
out vec4 vertexColor;
out vec2 texCoord0;
out float lmx;
out float lmy;
out vec4 glpos;

void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;
    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
    vertexDistance = length((ModelViewMat * vec4(Position, 1.0)).xyz);
    vertexColor = Color;
    texCoord0 = (TextureMat * vec4(UV0, 0.0, 1.0)).xy;
}
