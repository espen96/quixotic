#version 150
#moj_import <utils.glsl>

in vec3 Position;
in vec4 Color;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

out vec4 vertexColor;
in float lmx;
in float lmy;
in vec4 glpos;
void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);

    vertexColor = Color;
}
