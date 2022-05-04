#version 150
#extension GL_EXT_gpu_shader4_1 : enable
#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
in mat4 ProjMat2;

in vec4 vertexColor;
in vec2 texCoord0;
noperspective in vec3 test;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;
in vec3 cornerTex1;
in vec3 cornerTex2;
in vec3 cornerTex3;
in vec3 viewPos;

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

#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM

#define POM_DEPTH 0.1 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define MAX_ITERATIONS 50 // [5 10 15 20 25 30 40 50 60 70 80 90 100 125 150 200 400] //Improves quality at grazing angles (reduces performance)
#define MAX_DIST 30.0 // [5.0 10.0 15.0 20.0 25.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 125.0 150.0 200.0 400.0] //Increases distance at which POM is calculated

const float mincoord = 1.0/4096.0;
const float maxcoord = 1.0-mincoord;

const float MAX_OCCLUSION_DISTANCE = MAX_DIST;
const float MIX_OCCLUSION_DISTANCE = MAX_DIST*0.7;
const int   MAX_OCCLUSION_POINTS   = MAX_ITERATIONS;

#define POM_MAP_RES 128.0 // [16.0 32.0 64.0 128.0 256.0 512.0 1024.0] Increase to improve POM quality
const vec3 intervalMult = vec3(1.0, 1.0, 1.0/POM_DEPTH)/POM_MAP_RES * 1.0;


void main() {
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

/*
    // get view space normal from position derivative, 
    // since normals for flowing water in vertex, are not always surface aligned.
    vec3 vertex_normal = normalize(cross(dFdx(viewPos), dFdy(viewPos)));

    // get tbn matrix
    vec3 dp1 = dFdx(viewPos);
    vec3 dp2 = dFdy(viewPos);
    vec2 duv1 = dFdx(texCoord0);
    vec2 duv2 = dFdy(texCoord0);   // solve the linear system
    vec3 dp2perp = cross(dp2, vertex_normal);
    vec3 dp1perp = cross(vertex_normal, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;   // construct a scale-invariant frame
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    mat3 tbnMatrix = mat3(T * invmax, B * invmax, vertex_normal);
    vec2 cornerUV1 = cornerTex1.xy / cornerTex1.z;
    vec2 cornerUV2 = cornerTex2.xy / cornerTex2.z;
    vec2 cornerUV3 = cornerTex3.xy / cornerTex3.z;
    vec2 minUV = min(cornerUV1, min(cornerUV2, cornerUV3));
    vec2 maxUV = max(cornerUV1, max(cornerUV2, cornerUV3));
    // get offset direction from viewPos, with tbn matrix
    vec3 viewDirection = (normalize(viewPos) * tbnMatrix);
    viewDirection.xy = viewDirection.xy / (-viewDirection.z ) * 0.000005;
    vec2 texCoord0B = texCoord0;
    float lum0 = luma4(textureLod(Sampler0, texCoord0,100).rgb);

        for (float i = 0.0; luma4(albedo.rgb)/lum0*0.5 + i < 1 && i < MAX_OCCLUSION_POINTS; i+=0.01) {
        albedo = texture(Sampler0, texCoord0B);
        // use offset direction for offset
        texCoord0B.x += viewDirection.x;
        texCoord0B.y += viewDirection.y;
        // albedo.rgb =  (viewDirection);
        if (texCoord0B.x < minUV.x) {

            texCoord0B.x += maxUV.x - minUV.x;
        }
        if (texCoord0B.y < minUV.y) {
            texCoord0B.y += maxUV.y - minUV.y;
        }
        if (texCoord0B.x > maxUV.x) {
            texCoord0B.x += minUV.x - maxUV.x;
        }
        if (texCoord0B.y > maxUV.y) {
            texCoord0B.y += minUV.y - maxUV.y;
        }



    }
 */
vec4 color = albedo;

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
  if(res == 0.0f && !gui) {
alpha3 = alpha2;
color.b =  clamp(lmx, 0, 0.95);
color.r =  clamp(lmy, 0, 0.95);

}
fragColor = vec4(color.rgb,floor(map(alpha0, 0, 255, 0, 255)) / 255);


}
