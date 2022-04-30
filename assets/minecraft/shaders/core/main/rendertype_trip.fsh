#version 150
#extension GL_EXT_gpu_shader4_1 : enable
#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
in mat4 ProjMat2;

in vec4 vertexColor2;
in vec2 texCoord0;
noperspective in vec3 test;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;


void main() {


vec4 color = texture(Sampler0, texCoord0);



if(color.a * 255 <= 17.0) {
discard;
}
color.rgb = clamp(color.rgb, 0.001, 1);


fragColor = color;


}
