#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>
#extension GL_EXT_gpu_shader4_1 : enable

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
in vec4 vertexColor;

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

//vec3 rnd = clamp((vec3(fract(dither5x3() - dither64)))/8,0,1);
vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);

discardControlGLPos(gl_FragCoord.xy, glpos);

vec4 albedo = texture(Sampler0, texCoord0);

float atest = textureLod(Sampler0, texCoord0, 100).r;
float mipmapLevel = textureQueryLod(Sampler0, texCoord0).x;

//albedo.rgb = mix(albedo.rgb, test.rgb, clamp(mipmapLevel, 0, 1));

if(atest < 0.01) albedo =textureLod(Sampler0, texCoord0, 0); 
albedo.a = textureLod(Sampler0, texCoord0, 0).a;
//  float avgBlockLum = luma4(test*vertexColor.rgb * ColorModulator.rgb);
vec4 color = albedo * vertexColor * ColorModulator;
//  color.rgb = clamp(color.rgb*clamp(pow(avgBlockLum,-0.33)*0.85,-0.2,1.2),0.0,1.0);

float alpha = color.a;

//  color.rgb = (color.rgb*lm2.rgb);

color.rgb += rnd;
color.rgb = clamp(color.rgb, 0.01, 1);

float translucent = 0;

float mod2 = gl_FragCoord.x + gl_FragCoord.y;
float res = mod(mod2, 2.0f);

float lum = luma4(albedo.rgb);
vec3 diff = albedo.rgb - lum;

float alpha0 = int(textureLod(Sampler0, texCoord0, 0).a * 255);
float procedual1 = ((distance(textureLod(Sampler0, texCoord0, 0).rgb, test.rgb))) * 255;

if(alpha0 == 255) {
vec3 test2 = floor(test.rgb * 255);
float test3 = floor(test2.r + test2.g + test2.b);

}

float noise = luma4(rnd) * 128;

if(alpha0 >= sssMin && alpha0 <= sssMax) alpha0 = int(clamp(alpha0 + 0, sssMin, sssMax)); // SSS

if(alpha0 >= lightMin && alpha0 <= lightMax) alpha0 = int(clamp(alpha0 + (noise * 0.5), lightMin, lightMax)); // Emissives

if(alpha0 >= roughMin && alpha0 <= roughMax) alpha0 = int(clamp(alpha0 + noise, roughMin, roughMax)); // Roughness

if(alpha0 >= metalMin && alpha0 <= metalMax) alpha0 = int(clamp(alpha0 + noise, metalMin, metalMax)); // Metals

noise /= 255;

float alpha1 = 0.0;
float alpha2 = 0.0;

if(alpha0 <= 128) alpha1 = floor(map(alpha0, 0, 128, 0, 255)) / 255;
if(alpha0 >= 128) alpha2 = floor(map(alpha0, 128, 255, 0, 255)) / 255;

  //  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

float alpha3 = alpha1;
float lm = lmx + (luma4(rnd * clamp(lmx * 100, 0, 1)));
if(res == 0.0f) {
lm = lmy + (luma4(rnd * clamp(lmy * 100, 0, 1)));
alpha3 = alpha2;
}
fragColor = vec4(color.rgb,packUnorm2x4(alpha3, clamp(lm, 0, 0.95)));


}
