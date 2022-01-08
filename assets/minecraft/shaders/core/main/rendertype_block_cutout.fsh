#version 150
#extension GL_EXT_gpu_shader4_1 : enable
#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
in mat4 ProjMat2;
uniform vec3 ChunkOffset;
in vec4 vertexColor;
in vec2 texCoord0;
noperspective in vec3 test;
in vec4 glpos;
in float lmx;
in float lmy;

in vec4 normal;
in float dataFace;
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
float dither5x3() {
    const int ditherPattern[15] = int[15] (9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);

    int dither = ditherPattern[int(texCoord0.x) + int(texCoord0.y) * 5];

    return float(dither) * 0.0666666666666667f;
}

float dither64 = Bayer64(gl_FragCoord.xy);


void main() {
      if (dataFace < 0.5) {
    vec2 p = texCoord0+fract(GameTime/24000);

    bool gui = isGUI( ProjMat2);
vec3 rnd = clamp((vec3(fract(dither5x3() - dither64)))/8,0,1);
//vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);
rnd = vec3(bluenoise(p))/4;
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
float lightm = 0;

if(color.a * 255 <= 17.0) {
discard;
}
//color.rgb += rnd/255;
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


float noise = luma4(rnd) * 128;

if(alpha0 >= sssMin && alpha0 <= sssMax) alpha0 = int(clamp(alpha0 + 0 , sssMin, sssMax)); // SSS

if(alpha0 >= lightMin && alpha0 <= lightMax) alpha0 = int(clamp(alpha0 + 0, lightMin, lightMax)); // Emissives

if(alpha0 >= roughMin && alpha0 <= roughMax) alpha0 = int(clamp(alpha0 + 0, roughMin, roughMax)); // Roughness

if(alpha0 >= metalMin && alpha0 <= metalMax) alpha0 = int(clamp(alpha0 + 0, metalMin, metalMax)); // Metals

noise /= 255;

float alpha1 = 0.0;
float alpha2 = 0.0;

if(alpha0 <= 128) alpha1 = floor(map(alpha0, 0, 128, 0, 255)) / 255;
if(alpha0 >= 128) alpha2 = floor(map(alpha0, 128, 255, 0, 255)) / 255;

  //  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

float alpha3 = alpha1;
float lm = lmx;
  if(res == 0.0f && !gui) {
lm = lmy + ((rnd.x * clamp(lmy * 100, 0, 1)));
alpha3 = alpha2;
color.b =  clamp(lmx, 0, 0.95);
color.r =  clamp(lmy, 0, 0.95);

}
fragColor = vec4(color.rgb,floor(map(alpha0, 0, 255, 0, 255)) / 255);




    } else if (dataFace < 1.5) {
        fragColor = vertexColor;
    } else {
        vec3 storedChunkOffset = mod(ChunkOffset, vec3(16)) / 16.0;
        fragColor = vec4(encodeFloat(storedChunkOffset[int(gl_FragCoord.x)]), 1);
    }

}
