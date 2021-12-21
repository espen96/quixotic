#version 150

#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;
in ivec2 UV2;

uniform sampler2D Sampler2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform vec4 ColorModulator;

out float vertexDistance;
out vec4 vertexColor;
out vec4 glpos;
out float lmx;
out float lmy;
void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;
    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
//    vertexDistance = length((ModelViewMat * vec4(Position, 1.0)).xyz);
    vertexColor = Color * ColorModulator * minecraft_sample_lightmap(Sampler2, UV2);
}
