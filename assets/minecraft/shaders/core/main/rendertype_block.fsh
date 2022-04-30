#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>
#extension GL_EXT_gpu_shader4_1 : enable

uniform sampler2D Sampler0;
uniform sampler2D Sampler2;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
in vec4 vertexColor;
in mat4 ProjMat2;

noperspective in vec3 test;
in vec2 texCoord0;
in vec4 normal;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;


float dither5x3() {
    const int ditherPattern[15] = int[15] (9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);
    int dither = ditherPattern[int(texCoord0.x) + int(texCoord0.y) * 5];

    return float(dither) * 0.0666666666666667f;
}

float dither64 = Bayer64(gl_FragCoord.xy);


void main() {
    vec2 p = texCoord0+fract(GameTime/24000);

    bool gui = isGUI( ProjMat2);
discardControlGLPos(gl_FragCoord.xy, glpos);

vec4 albedo = texture(Sampler0, texCoord0);

float atest = textureLod(Sampler0, texCoord0, 100).r;
if(atest < 0.01) albedo =textureLod(Sampler0, texCoord0, 0); 
albedo.a = textureLod(Sampler0, texCoord0, 0).a;
vec4 color = albedo * vertexColor * ColorModulator;

color.rgb = clamp(color.rgb, 0.01, 1);


float mod2 = gl_FragCoord.x + gl_FragCoord.y;
float res = mod(mod2, 2.0f);


  if(res == 0.0f && !gui) {
color.b =  clamp(lmx, 0, 0.95);
color.r =  clamp(lmy, 0, 0.95);
}
fragColor = color;


}
