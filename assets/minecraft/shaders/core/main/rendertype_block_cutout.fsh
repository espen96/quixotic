#version 150
#extension GL_EXT_gpu_shader4_1 : enable
#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;

in vec4 vertexColor;
in vec2 texCoord0;
noperspective in vec3 test;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;

vec4 smoothfilter(in sampler2D tex, in vec2 uv) {
vec2 textureResolution = (textureSize(tex, 0).xy);
uv = uv * textureResolution + 0.5;
vec2 iuv = floor(uv);
vec2 fuv = fract(uv);
uv = iuv + fuv * fuv * fuv * (fuv * (fuv * 6.0 - 15.0) + 10.0);
uv = (uv - 0.5) / textureResolution;
return texture2D(tex, uv);
}

void main() {

vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);

discardControlGLPos(gl_FragCoord.xy, glpos);

vec4 albedo = textureLod(Sampler0, texCoord0, 0);
float mipmapLevel = textureQueryLod(Sampler0, texCoord0).x;

albedo.rgb = mix(albedo.rgb, test.rgb, clamp(mipmapLevel, 0, 1));

if(albedo.a > 0.5) albedo = texture(Sampler0, texCoord0);

//  float avgBlockLum = luma4(test*vertexColor.rgb * ColorModulator.rgb);
vec4 color = albedo * vertexColor * ColorModulator;
//  color.rgb = clamp(color.rgb*clamp(pow(avgBlockLum,-0.33)*0.85,-0.2,1.2),0.0,1.0);

float alpha = color.a;
float lightm = 0;

if(color.a * 255 <= 17.0) {
discard;
}
color.rgb += rnd / 16;
color.rgb = clamp(color.rgb, 0.001, 1);

float translucent = 0;

float mod2 = gl_FragCoord.x + gl_FragCoord.y;
float res = mod(mod2, 2.0f);

float lum = luma4(albedo.rgb);
vec3 diff = albedo.rgb - lum;

float alpha0 = int(textureLod(Sampler0, texCoord0, 0).a * 255);
if(alpha0 == 255) {
float procedual1 = ((distance(textureLod(Sampler0, texCoord0, 0).rgb, test.rgb))) * 255;

//color.rgb = test;
vec3 test2 = floor(test.rgb * 255);
float test3 = floor(test2.r + test2.g + test2.b);
//if(vertexColor.g >0.1)alpha0 =30; 
if(diff.r < 0.1 && diff.b < 0.05) alpha0 = int(floor(map((albedo.g * 0.1) * 255, 0, 255, sssMin, sssMax)));

 //if(test3 <= 305 && test3 >= 295 && test2.r >= 110 && test2.b <= 90)  alpha0 = clamp(procedual1*albedo.r,lightMin,lightMax);
 //if(test3 <= 255 && test3 >= 250 && test2.r >= 105 && test2.b <= 90)  alpha0 = clamp(procedual1*albedo.r,lightMin,lightMax);
}


float noise = luma4(rnd) * 255;

if(alpha0 >= sssMin && alpha0 <= sssMax) alpha0 = int(clamp(alpha0 + noise * 0.1, sssMin, sssMax)); // SSS

if(alpha0 >= lightMin && alpha0 <= lightMax) alpha0 = int(clamp(alpha0 + 0, lightMin, lightMax)); // Emissives

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
