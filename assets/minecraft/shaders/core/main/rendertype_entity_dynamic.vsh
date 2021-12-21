#version 150
#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in ivec2 UV2;

uniform sampler2D Sampler2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
out float lmx;
out float lmy;
out vec4 vertexColor;
out vec2 texCoord0;

out vec4 glpos;
void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;
    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
//    vertexDistance = length((ModelViewMat * vec4(Position, 1.0)).xyz);
    vertexColor = Color * minecraft_sample_lightmap2(Sampler2, UV2);
    texCoord0 = UV0;

}
