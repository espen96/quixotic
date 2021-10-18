#version 150

#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in ivec2 UV1;
in ivec2 UV2;
in vec3 Normal;
out float lm;
uniform sampler2D Sampler1;
uniform sampler2D Sampler2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

uniform vec3 Light0_Direction;
uniform vec3 Light1_Direction;
out float lmx;
out float lmy;
out mat4 ProjMat2;
out float vertexDistance;
out vec4 vertexColor;
out vec4 lightMapColor;
out vec4 overlayColor;
out vec2 texCoord0;
out vec4 normal;
out vec4 glpos;

void main() {
    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);
    glpos = gl_Position;

//    vertexDistance = length((ModelViewMat * vec4(Position, 1.0)).xyz);
    vertexColor = minecraft_mix_light(Light0_Direction, Light1_Direction, Normal, Color);
    lightMapColor = minecraft_sample_lightmap2(Sampler2, UV2);
    overlayColor = texelFetch(Sampler1, UV1, 0);
    texCoord0 = UV0;
    lm = clamp((float(UV2.y)/255)-(float(UV2.x)/255),0,1);
    ProjMat2 = ProjMat;

    normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);
    glpos = gl_Position;
        lmx = clamp((float(UV2.y)/255),0,1);
    lmy = clamp((float(UV2.x)/255),0,1);
}
