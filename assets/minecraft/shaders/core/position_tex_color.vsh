#version 150
#moj_import <projection.glsl>
in vec3 Position;
in vec2 UV0;
in vec4 Color;
out vec4 texProj0;
uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
out mat4 gbufferModelViewInverse;

out vec2 texCoord0;
out vec4 vertexColor;

void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    gbufferModelViewInverse = inverse(ProjMat * ModelViewMat);
    texProj0 = projection_from_position(gl_Position);

    texCoord0 = UV0;
    vertexColor = Color;
}
